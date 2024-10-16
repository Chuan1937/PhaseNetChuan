import argparse
import logging
import multiprocessing
import os
import pickle
import time
from functools import partial

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from data_reader import DataReader_mseed_array, DataReader_pred
from model import ModelConfig, UNet
from postprocess import (
    extract_amplitude,
    extract_picks,
    save_picks,
    save_picks_json,
    save_prob_h5,
)
from tqdm import tqdm
from visulization import plot_waveform

# 禁用TensorFlow的Eager Execution模式以提高执行效率
tf.compat.v1.disable_eager_execution()
# 设置TensorFlow的日志级别为ERROR，减少输出的信息量
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def read_args():
    """
    解析命令行参数的函数，用于获取用户输入的各种参数。
    
    Returns:
        argparse.Namespace: 命令行参数的Namespace对象，包含各个参数。
    """
    parser = argparse.ArgumentParser()
    
    # 添加各个命令行参数及其描述
    parser.add_argument("--batch_size", default=20, type=int, help="Batch size for prediction")
    parser.add_argument("--model_dir", help="Directory containing model checkpoints")
    parser.add_argument("--data_dir", default="", help="Directory of input data files")
    parser.add_argument("--data_list", default="", help="CSV file containing data file names")
    parser.add_argument("--hdf5_file", default="", help="Input HDF5 file path")
    parser.add_argument("--hdf5_group", default="data", help="Group name in HDF5 file")
    parser.add_argument("--result_dir", default="results", help="Directory to save results")
    parser.add_argument("--result_fname", default="picks", help="Name of the output file")
    parser.add_argument("--min_p_prob", default=0.3, type=float, help="Minimum probability for P-phase detection")
    parser.add_argument("--min_s_prob", default=0.3, type=float, help="Minimum probability for S-phase detection")
    parser.add_argument("--mpd", default=50, type=float, help="Minimum peak distance for detection")
    parser.add_argument("--amplitude", action="store_true", help="If specified, return amplitude values")
    parser.add_argument("--format", default="numpy", help="Format of input data")
    parser.add_argument("--s3_url", default="localhost:9000", help="S3 URL for storage")
    parser.add_argument("--stations", default="", help="Seismic station information")
    parser.add_argument("--plot_figure", action="store_true", help="If true, generate plots for results")
    parser.add_argument("--save_prob", action="store_true", help="If true, save prediction probabilities")
    parser.add_argument("--pre_sec", default=1, type=float, help="Window length before the pick")
    parser.add_argument("--post_sec", default=4, type=float, help="Window length after the pick")
    parser.add_argument("--highpass_filter", default=0.0, type=float, help="Apply a high-pass filter to data")
    parser.add_argument("--response_xml", default=None, type=str, help="Path to response XML file")
    parser.add_argument("--sampling_rate", default=100, type=float, help="Sampling rate of the data")
    
    # 解析并返回参数
    args = parser.parse_args()
    return args

def pred_fn(args, data_reader, figure_dir=None, prob_dir=None, log_dir=None):
    """
    预测函数，根据输入数据进行地震相位拾取。
    
    Args:
        args (argparse.Namespace): 命令行参数。
        data_reader (DataReader): 数据读取器实例，负责加载数据。
        figure_dir (str): 可选，保存图像的目录。
        prob_dir (str): 可选，保存概率结果的目录。
        log_dir (str): 可选，保存日志的目录。
    """
    # 获取当前时间，用于日志记录
    current_time = time.strftime("%y%m%d-%H%M%S")
    
    # 创建日志目录
    if log_dir is None:
        log_dir = os.path.join(args.log_dir, "pred", current_time)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 如果需要生成图像，创建图像保存目录
    if (args.plot_figure == True) and (figure_dir is None):
        figure_dir = os.path.join(log_dir, "figures")
        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)

    # 如果需要保存概率，创建保存概率的目录
    if (args.save_prob == True) and (prob_dir is None):
        prob_dir = os.path.join(log_dir, "probs")
        if not os.path.exists(prob_dir):
            os.makedirs(prob_dir)

    # 如果选择保存概率，创建HDF5文件来保存概率数据
    if args.save_prob:
        h5 = h5py.File(os.path.join(args.result_dir, "result.h5"), "w", libver="latest")
        prob_h5 = h5.create_group("/prob")

    # 记录日志信息
    logging.info("Pred log: %s" % log_dir)
    logging.info("Dataset size: {}".format(data_reader.num_data))

    # 创建TensorFlow数据管道，获取批次数据
    with tf.compat.v1.name_scope("Input_Batch"):
        if args.format == "mseed_array":
            batch_size = 1
        else:
            batch_size = args.batch_size
        dataset = data_reader.dataset(batch_size)
        batch = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

    # 加载模型配置
    config = ModelConfig(X_shape=data_reader.X_shape)

    # 保存模型配置到日志目录
    with open(os.path.join(log_dir, "config.log"), "w") as fp:
        fp.write("\n".join("%s: %s" % item for item in vars(config).items()))

    # 构建U-Net模型用于预测
    model = UNet(config=config, input_batch=batch, mode="pred")
    
    # TensorFlow会话配置
    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    # 启动TensorFlow会话
    with tf.compat.v1.Session(config=sess_config) as sess:
        # 恢复训练好的模型
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=5)
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)
        latest_check_point = tf.train.latest_checkpoint(args.model_dir)
        logging.info(f"restoring model {latest_check_point}")
        saver.restore(sess, latest_check_point)

        # 开始预测过程
        picks = []
        amps = [] if args.amplitude else None
        if args.plot_figure:
            multiprocessing.set_start_method("spawn")
            pool = multiprocessing.Pool(multiprocessing.cpu_count())

        for _ in tqdm(range(0, data_reader.num_data, batch_size), desc="Pred"):
            # 获取预测结果
            if args.amplitude:
                pred_batch, X_batch, amp_batch, fname_batch, t0_batch, station_batch = sess.run(
                    [model.preds, batch[0], batch[1], batch[2], batch[3], batch[4]],
                    feed_dict={model.drop_rate: 0, model.is_training: False},
                )
            else:
                pred_batch, X_batch, fname_batch, t0_batch, station_batch = sess.run(
                    [model.preds, batch[0], batch[1], batch[2], batch[3]],
                    feed_dict={model.drop_rate: 0, model.is_training: False},
                )

            waveforms = amp_batch if args.amplitude else None

            # 提取P和S相位拾取点
            picks_ = extract_picks(
                preds=pred_batch,
                file_names=fname_batch,
                station_ids=station_batch,
                begin_times=t0_batch,
                config=args,
                waveforms=waveforms,
                use_amplitude=args.amplitude,
                dt=1.0 / args.sampling_rate,
            )
            picks.extend(picks_)

            # 如果启用了图像生成，绘制波形图像
            if args.plot_figure:
                fname_batch = [x.decode() for x in fname_batch]
                pool.starmap(
                    partial(
                        plot_waveform,
                        figure_dir=figure_dir,
                    ),
                    zip(X_batch, pred_batch, fname_batch),
                )

            # 如果启用了保存概率功能，将结果保存到HDF5文件
            if args.save_prob:
                fname_batch = [x.decode() for x in fname_batch]
                save_prob_h5(pred_batch, fname_batch, prob_h5)

        # 保存最终的拾取结果到CSV文件
        if len(picks) > 0:
            df = pd.DataFrame(picks)
            base_columns = [
                "station_id",
                "begin_time",
                "phase_index",
                "phase_time",
                "phase_score",
                "phase_type",
                "file_name",
            ]
            if args.amplitude:
                base_columns.append("phase_amplitude")
                base_columns.append("phase_amp")
                df["phase_amp"] = df["phase_amplitude"]

            df = df[base_columns]
            df.to_csv(os.path.join(args.result_dir, args.result_fname + ".csv"), index=False)

            print(
                f"Done with {len(df[df['phase_type'] == 'P'])} P-picks and {len(df[df['phase_type'] == 'S'])} S-picks"
            )
        else:
            print(f"Done with 0 P-picks and 0 S-picks")

    return 0

def main(args):
    """
    主函数，负责初始化数据读取器并调用预测函数。
    
    Args:
        args (argparse.Namespace): 命令行参数。
    """
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

    # 创建输入数据读取器
    with tf.compat.v1.name_scope("create_inputs"):
        if args.format == "mseed_array":
            data_reader = DataReader_mseed_array(
                data_dir=args.data_dir,
                data_list=args.data_list,
                stations=args.stations,
                amplitude=args.amplitude,
                highpass_filter=args.highpass_filter,
            )
        else:
            data_reader = DataReader_pred(
                format=args.format,
                data_dir=args.data_dir,
                data_list=args.data_list,
                hdf5_file=args.hdf5_file,
                hdf5_group=args.hdf5_group,
                amplitude=args.amplitude,
                highpass_filter=args.highpass_filter,
                response_xml=args.response_xml,
                sampling_rate=args.sampling_rate,
            )

        # 调用预测函数进行处理
        pred_fn(args, data_reader, log_dir=args.result_dir)

    return

# 入口点：从命令行读取参数并执行主函数
if __name__ == "__main__":
    args = read_args()
    main(args)
