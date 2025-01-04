# Setting Up the Development Environment

Currently, this project is built and run in a Docker container based on image **python:3.8.20**.

I set up the python environment with these commands

        python -m venv .venv

        source .venv/bin/activate

        pip install -r requirements.txt

```bash
ofenloch@358b4b78a261:~/workspaces/time-series-forecasting-pytorch-ofenloch$ which python
/usr/local/bin/python
ofenloch@358b4b78a261:~/workspaces/time-series-forecasting-pytorch-ofenloch$ python --version
Python 3.8.20
ofenloch@358b4b78a261:~/workspaces/time-series-forecasting-pytorch-ofenloch$
ofenloch@358b4b78a261:~/workspaces/time-series-forecasting-pytorch-ofenloch$ python -m venv .venv
ofenloch@358b4b78a261:~/workspaces/time-series-forecasting-pytorch-ofenloch$ source .venv/bin/activate
(.venv) ofenloch@358b4b78a261:~/workspaces/time-series-forecasting-pytorch-ofenloch$
(.venv) ofenloch@358b4b78a261:~/workspaces/time-series-forecasting-pytorch-ofenloch$ which python
/home/ofenloch/workspaces/time-series-forecasting-pytorch-ofenloch/.venv/bin/python
(.venv) ofenloch@358b4b78a261:~/workspaces/time-series-forecasting-pytorch-ofenloch$ python --version
Python 3.8.20
(.venv) ofenloch@358b4b78a261:~/workspaces/time-series-forecasting-pytorch-ofenloch$
(.venv) ofenloch@358b4b78a261:~/workspaces/time-series-forecasting-pytorch-ofenloch$ more requirements.txt
numpy
torch
alpha_vantage
matplotlib
(.venv) ofenloch@358b4b78a261:~/workspaces/time-series-forecasting-pytorch-ofenloch$ pip install -r requirements.txt
Collecting numpy
  Downloading numpy-1.24.4-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (17.3 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 17.3/17.3 MB 20.8 MB/s eta 0:00:00
Collecting torch
  Downloading torch-2.4.1-cp38-cp38-manylinux1_x86_64.whl (797.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 797.1/797.1 MB 1.9 MB/s eta 0:00:00
Collecting alpha_vantage
  Downloading alpha_vantage-3.0.0-py3-none-any.whl (35 kB)
Collecting matplotlib
  Downloading matplotlib-3.7.5-cp38-cp38-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (9.2 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 9.2/9.2 MB 23.5 MB/s eta 0:00:00
Collecting typing-extensions>=4.8.0
  Downloading typing_extensions-4.12.2-py3-none-any.whl (37 kB)
Collecting jinja2
  Downloading jinja2-3.1.5-py3-none-any.whl (134 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 134.6/134.6 kB 3.9 MB/s eta 0:00:00
Collecting nvidia-nvtx-cu12==12.1.105
  Downloading nvidia_nvtx_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (99 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 99.1/99.1 kB 3.2 MB/s eta 0:00:00
Collecting nvidia-nccl-cu12==2.20.5
  Downloading nvidia_nccl_cu12-2.20.5-py3-none-manylinux2014_x86_64.whl (176.2 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 176.2/176.2 MB 6.7 MB/s eta 0:00:00
Collecting nvidia-cudnn-cu12==9.1.0.70
  Downloading nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl (664.8 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 664.8/664.8 MB 1.9 MB/s eta 0:00:00
Collecting nvidia-cusparse-cu12==12.1.0.106
  Downloading nvidia_cusparse_cu12-12.1.0.106-py3-none-manylinux1_x86_64.whl (196.0 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 196.0/196.0 MB 6.4 MB/s eta 0:00:00
Collecting nvidia-cublas-cu12==12.1.3.1
  Downloading nvidia_cublas_cu12-12.1.3.1-py3-none-manylinux1_x86_64.whl (410.6 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 410.6/410.6 MB 3.6 MB/s eta 0:00:00
Collecting nvidia-cusolver-cu12==11.4.5.107
  Downloading nvidia_cusolver_cu12-11.4.5.107-py3-none-manylinux1_x86_64.whl (124.2 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 124.2/124.2 MB 8.4 MB/s eta 0:00:00
Collecting nvidia-cuda-runtime-cu12==12.1.105
  Downloading nvidia_cuda_runtime_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (823 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 823.6/823.6 kB 10.9 MB/s eta 0:00:00
Collecting filelock
  Downloading filelock-3.16.1-py3-none-any.whl (16 kB)
Collecting nvidia-cuda-nvrtc-cu12==12.1.105
  Downloading nvidia_cuda_nvrtc_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (23.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 23.7/23.7 MB 18.9 MB/s eta 0:00:00
Collecting nvidia-cufft-cu12==11.0.2.54
  Downloading nvidia_cufft_cu12-11.0.2.54-py3-none-manylinux1_x86_64.whl (121.6 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 121.6/121.6 MB 9.0 MB/s eta 0:00:00
Collecting nvidia-curand-cu12==10.3.2.106
  Downloading nvidia_curand_cu12-10.3.2.106-py3-none-manylinux1_x86_64.whl (56.5 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 56.5/56.5 MB 13.7 MB/s eta 0:00:00
Collecting nvidia-cuda-cupti-cu12==12.1.105
  Downloading nvidia_cuda_cupti_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (14.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 14.1/14.1 MB 22.9 MB/s eta 0:00:00
Collecting triton==3.0.0
  Downloading triton-3.0.0-1-cp38-cp38-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (209.4 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 209.4/209.4 MB 6.3 MB/s eta 0:00:00
Collecting fsspec
  Downloading fsspec-2024.12.0-py3-none-any.whl (183 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 183.9/183.9 kB 5.2 MB/s eta 0:00:00
Collecting sympy
  Downloading sympy-1.13.3-py3-none-any.whl (6.2 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.2/6.2 MB 22.7 MB/s eta 0:00:00
Collecting networkx
  Downloading networkx-3.1-py3-none-any.whl (2.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 14.8 MB/s eta 0:00:00
Collecting nvidia-nvjitlink-cu12
  Downloading nvidia_nvjitlink_cu12-12.6.85-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl (19.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 19.7/19.7 MB 20.2 MB/s eta 0:00:00
Collecting requests
  Downloading requests-2.32.3-py3-none-any.whl (64 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 64.9/64.9 kB 1.7 MB/s eta 0:00:00
Collecting aiohttp
  Downloading aiohttp-3.10.11-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.3 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.3/1.3 MB 16.6 MB/s eta 0:00:00
Collecting contourpy>=1.0.1
  Downloading contourpy-1.1.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (301 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 301.1/301.1 kB 6.1 MB/s eta 0:00:00
Collecting cycler>=0.10
  Downloading cycler-0.12.1-py3-none-any.whl (8.3 kB)
Collecting fonttools>=4.22.0
  Downloading fonttools-4.55.3-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.7/4.7 MB 21.8 MB/s eta 0:00:00
Collecting pillow>=6.2.0
  Downloading pillow-10.4.0-cp38-cp38-manylinux_2_28_x86_64.whl (4.5 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.5/4.5 MB 20.9 MB/s eta 0:00:00
Collecting pyparsing>=2.3.1
  Downloading pyparsing-3.1.4-py3-none-any.whl (104 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 104.1/104.1 kB 2.7 MB/s eta 0:00:00
Collecting kiwisolver>=1.0.1
  Downloading kiwisolver-1.4.7-cp38-cp38-manylinux_2_5_x86_64.manylinux1_x86_64.whl (1.2 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 11.7 MB/s eta 0:00:00
Collecting packaging>=20.0
  Downloading packaging-24.2-py3-none-any.whl (65 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 65.5/65.5 kB 1.9 MB/s eta 0:00:00
Collecting importlib-resources>=3.2.0
  Downloading importlib_resources-6.4.5-py3-none-any.whl (36 kB)
Collecting python-dateutil>=2.7
  Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 229.9/229.9 kB 5.1 MB/s eta 0:00:00
Collecting zipp>=3.1.0
  Downloading zipp-3.20.2-py3-none-any.whl (9.2 kB)
Collecting six>=1.5
  Downloading six-1.17.0-py2.py3-none-any.whl (11 kB)
Collecting aiosignal>=1.1.2
  Downloading aiosignal-1.3.1-py3-none-any.whl (7.6 kB)
Collecting aiohappyeyeballs>=2.3.0
  Downloading aiohappyeyeballs-2.4.4-py3-none-any.whl (14 kB)
Collecting attrs>=17.3.0
  Downloading attrs-24.3.0-py3-none-any.whl (63 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 63.4/63.4 kB 2.3 MB/s eta 0:00:00
Collecting frozenlist>=1.1.1
  Downloading frozenlist-1.5.0-cp38-cp38-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (243 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 243.1/243.1 kB 5.8 MB/s eta 0:00:00
Collecting async-timeout<6.0,>=4.0
  Downloading async_timeout-5.0.1-py3-none-any.whl (6.2 kB)
Collecting yarl<2.0,>=1.12.0
  Downloading yarl-1.15.2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (319 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 319.0/319.0 kB 5.1 MB/s eta 0:00:00
Collecting multidict<7.0,>=4.5
  Downloading multidict-6.1.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (129 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 129.7/129.7 kB 2.4 MB/s eta 0:00:00
Collecting MarkupSafe>=2.0
  Downloading MarkupSafe-2.1.5-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (26 kB)
Collecting urllib3<3,>=1.21.1
  Downloading urllib3-2.2.3-py3-none-any.whl (126 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 126.3/126.3 kB 3.4 MB/s eta 0:00:00
Collecting charset-normalizer<4,>=2
  Downloading charset_normalizer-3.4.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (144 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 144.7/144.7 kB 2.4 MB/s eta 0:00:00
Collecting idna<4,>=2.5
  Downloading idna-3.10-py3-none-any.whl (70 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 70.4/70.4 kB 1.8 MB/s eta 0:00:00
Collecting certifi>=2017.4.17
  Downloading certifi-2024.12.14-py3-none-any.whl (164 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 164.9/164.9 kB 4.0 MB/s eta 0:00:00
Collecting mpmath<1.4,>=1.1.0
  Downloading mpmath-1.3.0-py3-none-any.whl (536 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 536.2/536.2 kB 9.4 MB/s eta 0:00:00
Collecting propcache>=0.2.0
  Downloading propcache-0.2.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (213 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 213.6/213.6 kB 5.6 MB/s eta 0:00:00
Installing collected packages: mpmath, zipp, urllib3, typing-extensions, sympy, six, pyparsing, propcache, pillow, packaging, nvidia-nvtx-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-curand-cu12, nvidia-cufft-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, numpy, networkx, MarkupSafe, kiwisolver, idna, fsspec, frozenlist, fonttools, filelock, cycler, charset-normalizer, certifi, attrs, async-timeout, aiohappyeyeballs, triton, requests, python-dateutil, nvidia-cusparse-cu12, nvidia-cudnn-cu12, multidict, jinja2, importlib-resources, contourpy, aiosignal, yarl, nvidia-cusolver-cu12, matplotlib, torch, aiohttp, alpha_vantage
Successfully installed MarkupSafe-2.1.5 aiohappyeyeballs-2.4.4 aiohttp-3.10.11 aiosignal-1.3.1 alpha_vantage-3.0.0 async-timeout-5.0.1 attrs-24.3.0 certifi-2024.12.14 charset-normalizer-3.4.1 contourpy-1.1.1 cycler-0.12.1 filelock-3.16.1 fonttools-4.55.3 frozenlist-1.5.0 fsspec-2024.12.0 idna-3.10 importlib-resources-6.4.5 jinja2-3.1.5 kiwisolver-1.4.7 matplotlib-3.7.5 mpmath-1.3.0 multidict-6.1.0 networkx-3.1 numpy-1.24.4 nvidia-cublas-cu12-12.1.3.1 nvidia-cuda-cupti-cu12-12.1.105 nvidia-cuda-nvrtc-cu12-12.1.105 nvidia-cuda-runtime-cu12-12.1.105 nvidia-cudnn-cu12-9.1.0.70 nvidia-cufft-cu12-11.0.2.54 nvidia-curand-cu12-10.3.2.106 nvidia-cusolver-cu12-11.4.5.107 nvidia-cusparse-cu12-12.1.0.106 nvidia-nccl-cu12-2.20.5 nvidia-nvjitlink-cu12-12.6.85 nvidia-nvtx-cu12-12.1.105 packaging-24.2 pillow-10.4.0 propcache-0.2.0 pyparsing-3.1.4 python-dateutil-2.9.0.post0 requests-2.32.3 six-1.17.0 sympy-1.13.3 torch-2.4.1 triton-3.0.0 typing-extensions-4.12.2 urllib3-2.2.3 yarl-1.15.2 zipp-3.20.2

[notice] A new release of pip is available: 23.0.1 -> 24.3.1
[notice] To update, run: pip install --upgrade pip
(.venv) ofenloch@358b4b78a261:~/workspaces/time-series-forecasting-pytorch-ofenloch$
```


# tl;dr

Python is "Python 3.8.20"

Initial / original requirements were

        numpy
        torch
        alpha_vantage
        matplotlib

This was installed in my local **.venv** with `pip install -r requirements.txt`.

File **requirements.txt** was generated with `pip freeze > requirements.txt`.

File **requirements-pipreqs.txt** was generated with `pipreqs --savepath requirements-pipreqs.txt`.