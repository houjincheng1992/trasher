## 所遇问题

Traceback (most recent call last):
  File "digit-recognizer-example.py", line 143, in <module>
    sess.run(train_step_1,feed_dict={x:batch_x,y:batch_y,keep_prob:0.5})
  File "E:\software_env\anaconda\envs\tensorflow\lib\site-packages\tensorflow\python\client\session.py", line 950, in run
    run_metadata_ptr)
  File "E:\software_env\anaconda\envs\tensorflow\lib\site-packages\tensorflow\python\client\session.py", line 1173, in _run
    feed_dict_tensor, options, run_metadata)
  File "E:\software_env\anaconda\envs\tensorflow\lib\site-packages\tensorflow\python\client\session.py", line 1350, in _do_run
    run_metadata)
  File "E:\software_env\anaconda\envs\tensorflow\lib\site-packages\tensorflow\python\client\session.py", line 1356, in _do_call
    return fn(*args)
  File "E:\software_env\anaconda\envs\tensorflow\lib\site-packages\tensorflow\python\client\session.py", line 1341, in _run_fn
    options, feed_dict, fetch_list, target_list, run_metadata)
  File "E:\software_env\anaconda\envs\tensorflow\lib\site-packages\tensorflow\python\client\session.py", line 1429, in _call_tf_sessionrun
    run_metadata)
TypeError: TF_SessionRun_wrapper: expected all values in input dict to be ndarray

## 相关代码

* 相关代码参考 digit-recognizer-example.py

## 相关环境

1. conda 4.6.11
2. python 3.5.6
3. tensorflow-gpu 1.14.0
4. numpy 1.16.4
5. pandas 0.23.4
6. matplotlib 2.0.0
7. CUDA 10.0.140
8. cudnn 7.6.1