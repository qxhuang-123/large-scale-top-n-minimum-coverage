# TRMNL F 五个数据集本机运行

本机配置日期：2026-10-18。F仍是临时版本名；算法固定调用交付包的`state_source`模式。

## 在PyCharm中运行

打开现有项目`E:/PythonProject`，在运行配置下拉菜单选择：

- `TRMNL-OP-F`
- `TRMNL-YELP-F`
- `TRMNL-SO-F`
- `TRMNL-VG-F`
- `TRMNL-TG-F`

配置已保存到父项目的`.run/`目录，使用现有解释器：

`E:/PythonProject/.venv/Scripts/python.exe`（Python 3.9.13）。

也可以在项目树中右键运行本目录下同名的Python文件。当前命名仅是用户指定的运行名称，未修改算法数学名称或原有论文。

默认每个入口运行一次完整目标：N=10，OP请求覆盖2367，Yelp请求覆盖7017。运行参数可改为`--repeats 3`重复三次，或用`--target 具体整数`指定较低目标。PyCharm运行菜单未刷新时重新打开`E:/PythonProject`即可。

## 输出位置

- `outputs/TRMNL-OP-F/<时间>/`
- `outputs/TRMNL-YELP-F/<时间>/`

每次运行保留逐次JSON、推荐列表、已选边、覆盖证书（若适用）、计时CSV和汇总JSON；`latest_summary.json`指向该入口最近一次完整运行的汇总内容。

三个时间口径分开报告：

1. `call_wall_seconds`：完整C++核心调用，包括初始化及返回值转换。
2. `solve_wall_seconds`：输入再次校验、核心调用、推荐方案审计等Python接口工作。
3. `first_run_with_load_checks_and_export_seconds`：本次首次加载CSR、校验、求解、审计和输出的累计时间；不包含首次从XLSX生成缓存。

不要把首次XLSX转换、三次重复运行的总耗时或原始数据读取时间与PDF中的约10秒核心时间混用。

## Windows适配

原交付包只配置了Linux/Clang/GCC构建。本机增加`src/msvc_compat.h`，把两个位运算builtin映射到MSVC等价intrinsic；`setup.py`增加Windows C++17编译参数。算法循环、局部修复、源端/终点聚合和两个`.inc`文件未修改；`f_algorithm.py`和`run_f.py`也保持交付版本。

原始`setup.py`和三个核心源文件保存在`original_sources/`。原始SHA清单保留在`MANIFEST_SHA256.json`，本机适配与实际构建文件的哈希另存`outputs/local_build_manifest.json`。

重新编译：

```powershell
& 'E:/PythonProject/.venv/Scripts/python.exe' setup.py build_ext --inplace
```

无需安装WSL，也未修改现有Python环境中的包。交付说明建议Python 3.10以上；本机现有3.9解释器已通过40个小图、280个目标的独立LP对照，及本次OP/Yelp实跑。

## 数据来源与候选构造

OP使用`E:/Users/24qxh/Desktop/OP/IUIC1_Rui_pred_user_item_FULL_display1.xlsx`。Yelp使用论文目录中的`Yelp_R_ui_UNKNOWN_ONLY_known_ratings_0_display1.xlsx`，其中0表示已知/禁用对。两者均先转float32，与已有运行脚本的分数口径一致，再将核心评分数组存为float64。

本机没有找到原来的OP/Yelp候选缓存。因此按原脚本的`seed=20260704`、float32小扰动及argpartition方式重建40%候选，随后将保留行按原始评分降序、物品ID升序重排，以满足F的输入约定。OP使用完整评分候选，而Yelp使用未知评分候选；未擅自把两套输入统一成另一种预处理。

重建后的边数、候选物品数、初始覆盖和初始总分与归档记录一致；没有原缓存逐字节哈希，不能仅据这些汇总指标证明每条边逐字节相同。本次缓存和输入源记录在`data/OP/metadata.json`及`data/YELP/metadata.json`，后续运行直接复用它们。

首次重建入口：

```powershell
& 'E:/PythonProject/.venv/Scripts/python.exe' prepare_datasets.py OP
& 'E:/PythonProject/.venv/Scripts/python.exe' prepare_datasets.py YELP
```

已存在完整缓存时，程序直接复用；不会重新修改原始XLSX。

## SO / VG / TG 新增入口（2026-10-18）

三个入口使用同一个已编译的 F 核心和现有 Python 解释器。默认 N=10、运行一次，覆盖请求为原缓存中的全部候选物品：

| 运行配置 | 用户数 | 物品数 | 候选边数 | 默认覆盖请求 D |
| --- | ---: | ---: | ---: | ---: |
| TRMNL-SO-F | 35,598 | 18,357 | 261,282,620 | 17,951 |
| TRMNL-VG-F | 24,442 | 10,350 | 101,111,454 | 10,014 |
| TRMNL-TG-F | 19,412 | 11,924 | 92,529,265 | 11,709 |

这些 D 是请求值，是否可行由本次 F 求解及证书审计判断，不能预先当作最大可达覆盖。

复用的原候选缓存位于 `E:/PythonProject/.venv/NIPT_预测结果/<数据集>贪心算法_cache/<数据集>_cand_pos_top40_seed20260704_v2.npz`。转换直接保留原候选成员和评分，仅将每个用户的行按评分降序、物品 ID 升序重排，并建立 F 要求的边索引及反向数组，不重新筛选候选，不读取或转置 SO 的 CSV。原缓存只读。

首次点击运行会先转换格式，显示每 1000 个用户的转换进度，随后开始求解。八个本地数组约占 SO 7.79 GiB、VG 3.01 GiB、TG 2.76 GiB 磁盘空间；以后复用 `data/SO`、`data/VG`、`data/TG`。转换时间单独记录在 `candidate_metadata.preparation_seconds`，不算核心求解时间。建议逐个运行。

三个入口沿用 `--repeats 3`、`--target 整数` 参数；`--check-inputs` 仅核对原缓存头部、维度和指针，不进行转换或求解。输出保存到 `outputs/TRMNL-SO-F`、`outputs/TRMNL-VG-F`、`outputs/TRMNL-TG-F`，格式与 OP/Yelp 相同。

本次配置已通过入口导入、语法、运行配置 XML、原缓存维度/边数/候选物品数核查，以及微型数据的格式转换与双向 CSR 一致性检查。SO/VG/TG 的完整转换与求解留给用户运行；未填写未经实跑的速度或目标值。转换完成后会检查初始覆盖和总分与归档一致，求解结果仍执行 F 原有的推荐方案及覆盖证书审计。

### Windows 缓存占用修复

Windows 下，已被内存映射占用的 NPY 文件不能以 `w+` 重新创建，会报 `OSError: [Errno 22] Invalid argument`。准备阶段现已增加每个数据集的进程锁，重复启动时会明确提示另一个转换尚在运行；异常退出时也会显式释放已创建的映射。

为兼容旧进程或 Python Console 仍然持有的映射，每次需要转换时使用独立的 `data/<数据集>/arrays_<唯一编号>/` 目录，检查完成后才原子写入 `metadata.json`。入口通过其中的 `array_directory` 加载完成的数据，不再覆盖失败运行遗留的同名数组。旧文件和原始候选缓存均不删除。`preparation.lock` 文件可以长期保留，是否占用由系统锁判断，不要手动删除它。

修复已检查跨进程互斥、异常后重新转换、旧数组仍被映射时完成新一代转换，以及完成缓存复用。SO 此次另外执行实际规模的准备验证；没有调用正式求解。
