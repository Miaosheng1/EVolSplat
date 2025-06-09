python mono/tools/test_scale_cano.py \
    'mono/configs/HourglassDecoder/test_kitti_convlarge.0.3_150.py' \
    --load-from /data0/jxhuang/models/metric3d/convlarge_hourglass_0.3_150_step750k_v1.1.pth \
    --test_data_path /data0/jxhuang/datasets/multiscene_kitti/40/seq_00_nerfacto_0295_40 \
    --launcher None