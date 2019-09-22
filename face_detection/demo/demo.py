"""LFFD Demo."""
import os, sys
import argparse
import cv2
import time
import mxnet as mx
import numpy as np

sys.path.append("..")
from accuracy_evaluation import predict


def parse_args():
    parser = argparse.ArgumentParser(description='LFFD Demo.')
    parser.add_argument('--version', type=str, default='v2',
                        help='The version of pretrained model, now support "v1" and "v2".')
    parser.add_argument('--mode', type=str, default='image',
                        help='The format of input data, now support "image" of jpg and "video" of mp4.')
    parser.add_argument('--use-gpu', type=bool, default=False,
                        help='Default is cpu.')
    parser.add_argument('--data', type=str, default='./data',
                        help='The path of input and output file.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # context list
    if args.use_gpu:
        ctx = mx.gpu(0)
    else:
        ctx = mx.cpu()

    if args.version == 'v1':
        from config_farm import configuration_10_320_20L_5scales_v1 as cfg

        symbol_file_path = '../symbol_farm/symbol_10_560_25L_8scales_v1_deploy.json'
        model_file_path = '../saved_model/configuration_10_560_25L_8scales_v1/train_10_560_25L_8scales_v1_iter_1400000.params'
    elif args.version == 'v2':
        from config_farm import configuration_10_320_20L_5scales_v2 as cfg

        symbol_file_path = '../symbol_farm/symbol_10_320_20L_5scales_v2_deploy.json'
        model_file_path = '../saved_model/configuration_10_320_20L_5scales_v2/train_10_320_20L_5scales_v2_iter_1800000.params'
    else:
        raise TypeError('Unsupported LFFD Version.')

    face_predictor = predict.Predict(mxnet=mx,
                                     symbol_file_path=symbol_file_path,
                                     model_file_path=model_file_path,
                                     ctx=ctx,
                                     receptive_field_list=cfg.param_receptive_field_list,
                                     receptive_field_stride=cfg.param_receptive_field_stride,
                                     bbox_small_list=cfg.param_bbox_small_list,
                                     bbox_large_list=cfg.param_bbox_large_list,
                                     receptive_field_center_start=cfg.param_receptive_field_center_start,
                                     num_output_scales=cfg.param_num_output_scales)

    if args.mode == 'image':
        data_folder = args.data
        file_name_list = [file_name for file_name in os.listdir(data_folder) \
                          if file_name.lower().endswith('jpg')]

        for file_name in file_name_list:
            im = cv2.imread(os.path.join(data_folder, file_name))

            bboxes, infer_time = face_predictor.predict(im, resize_scale=1, score_threshold=0.6, top_k=10000, \
                                                        NMS_threshold=0.4, NMS_flag=True, skip_scale_branch_list=[])

            for bbox in bboxes:
                cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

            # if max(im.shape[:2]) > 1600:
            #     scale = 1600/max(im.shape[:2])
            #     im = cv2.resize(im, (0, 0), fx=scale, fy=scale)
            cv2.imshow('im', im)
            cv2.waitKey(5000)
            cv2.imwrite(os.path.join(data_folder, file_name.replace('.jpg', '_result.png')), im)
    elif args.mode == 'video':
        # win_name = 'LFFD DEMO'
        # cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        data_folder = args.data
        file_name_list = [file_name for file_name in os.listdir(data_folder) \
                          if file_name.lower().endswith('mp4')]
        for file_name in file_name_list:
            out_file = os.path.join(data_folder, file_name.replace('.mp4', '_v2_gpu_result.avi'))
            cap = cv2.VideoCapture(os.path.join(data_folder, file_name))
            vid_writer = cv2.VideoWriter(out_file, \
                                         cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 60, \
                                         (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), \
                                          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            while cv2.waitKey(1) < 0:
                ret, frame = cap.read()
                if ret:
                    h, w, c = frame.shape

                if not ret:
                    print("Done processing of %s" % file_name)
                    print("Output file is stored as %s" % out_file)
                    cv2.waitKey(3000)
                    break

                tic = time.time()
                bboxes, infer_time = face_predictor.predict(frame, resize_scale=1, score_threshold=0.6, top_k=10000, \
                                                            NMS_threshold=0.4, NMS_flag=True, skip_scale_branch_list=[])
                toc = time.time()
                detect_time = (toc - tic) * 1000

                face_num = 0
                for bbox in bboxes:
                    face_num += 1
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

                computing_platform = 'Computing platform: NVIDIA GPU FP32'
                cv2.putText(frame, computing_platform, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                input_resolution = 'Network input resolution: %sx%s' % (w, h)
                cv2.putText(frame, input_resolution, (5, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                infer_time_info = 'Inference time: %.2f ms' % (infer_time)
                cv2.putText(frame, infer_time_info, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                infer_speed = 'Inference speed: %.2f FPS' % (1000 / infer_time)
                cv2.putText(frame, infer_speed, (5, 135), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                face_num_info = 'Face num: %d' % (face_num)
                cv2.putText(frame, face_num_info, (5, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                vid_writer.write(frame.astype(np.uint8))
                # cv2.imshow(win_name, frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
    else:
        raise TypeError('Unsupported File Format.')


if __name__ == '__main__':
    main()
