""" このコードは関数シグネチャの例であり、実行は想定していません。
"""

torch.utils.tensorboard.writer.SummaryWriter(log_dir=None, comment='',
    purge_step=None, max_queue=10, flush_secs=120, filename_suffix='')
add_scalar(tag, scalar_value, global_step=None, walltime=None)
add_scalars(main_tag, tag_scalar_dict, global_step=None, walltime=None)
add_histogram(tag, values, global_step=None, bins='tensorflow', walltime=None, max_bins=None)
add_image(tag, img_tensor, global_step=None, walltime=None,  dataformats='CHW')
add_images(tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW')
add_figure(tag, figure, global_step=None, close=True, walltime=None)
add_video(tag, vid_tensor, global_step=None, fps=4, walltime=None)
add_audio(tag, snd_tensor, global_step=None, sample_rate=44100,
    walltime=None)
add_text(tag, text_string, global_step=None, walltime=None)
add_graph(model, input_to_model=None, verbose=False)
add_embedding(mat, metadata=None, label_img=None, global_step=None,
    tag='default', metadata_header=None)
add_pr_curve(tag, labels, predictions, global_step=None, num_thresholds=127,
    weights=None, walltime=None)
