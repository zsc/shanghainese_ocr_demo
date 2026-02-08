# 上海方言词典 OCR 处理工具

这是一个用于处理《上海方言词典》PDF 页面的 Python 工具集，能够自动检测页面布局、分割左右两栏、提取文字行并保存为独立图像，便于后续的 OCR 文字识别和数字化处理。

## 功能特性

- **自动页面分割**：检测页面中间的竖线，自动将页面分为左栏和右栏
- **高精度文字检测**：使用连通域分析（Connected Components）检测文字，高召回率
- **智能行合并**：基于 Y 坐标聚类，将同一行的文字块合并为完整的长条图像
- **批量处理**：支持批量处理多页 PDF
- **可视化调试**：生成带编号的可视化图像，便于检查结果

## 安装依赖

```bash
pip install PyMuPDF opencv-python-headless numpy Pillow
```

依赖说明：
- `PyMuPDF` (fitz)：PDF 页面提取
- `opencv-python-headless`：图像处理和连通域检测
- `numpy`：数值计算
- `Pillow`：图像保存和可视化

## 项目结构

```
.
├── README.md                      # 本文件
├── .gitignore                     # Git 忽略配置
├── 上海方言词典.pdf                # 原始词典 PDF（未提交）
├── first_page.png                 # 示例：第一页图像
├── process_dictionary_page.py     # 单页处理核心脚本
├── batch_process.py               # 批量处理脚本
└── extracted/                     # 提取结果目录（自动生成）
    ├── page_01/
    │   ├── debug_visualization.png    # 可视化调试图
    │   ├── box_001_left_01.png        # 提取的文字行图像
    │   ├── box_002_left_02.png
    │   └── ...
    ├── page_02/
    └── ...
```

## 使用方法

### 1. 单页处理

处理单张页面图像：

```python
from process_dictionary_page import process_dictionary_page

# 处理第一页
result_paths = process_dictionary_page(
    image_path="first_page.png",
    output_dir="extracted/page_01"
)

print(f"提取了 {len(result_paths)} 个文字行")
```

### 2. 批量处理前十页

```bash
python3 batch_process.py
```

这将：
1. 自动提取 PDF 的前十页为图像
2. 逐页处理（检测竖线 → 提取文字框 → 聚类合并 → 保存图像）
3. 生成每页的可视化调试图和提取的文字行

### 3. 处理自定义页面

```python
import fitz  # PyMuPDF
from process_dictionary_page import process_dictionary_page

# 打开 PDF
doc = fitz.open("上海方言词典.pdf")

# 提取并处理特定页面
page_num = 5
page = doc[page_num]
pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
pix.save(f"page_{page_num}.png")

result = process_dictionary_page(
    f"page_{page_num}.png",
    f"extracted/page_{page_num}"
)

doc.close()
```

## 算法流程

### 1. 竖线检测与页面分割

```python
def detect_vertical_line(image):
    # 使用霍夫变换检测页面中间的竖线
    # 返回竖线的 x 坐标，用于分割左右两栏
```

基于页面布局特点（双栏排版），自动检测中间分隔线，将页面分为左右两部分独立处理。

### 2. 文字检测（高召回率）

```python
def detect_text_boxes_raw(image):
    # 1. 二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 2. 轻微膨胀连接字符
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.dilate(binary, kernel, iterations=1)
    
    # 3. 连通域检测（8-连通）
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
```

使用 OpenCV 的连通域检测算法，能够检测到非常小的文字块（包括 IPA 音标），确保高召回率。

### 3. Y 坐标聚类与行合并

```python
def cluster_by_y(boxes, y_tolerance=12):
    # 按 Y 中心坐标排序
    # 将 Y 差值小于阈值的框分到同一组
    # 合并同一组内的所有框为一个大框
```

核心思想：同一行的文字具有相似的 Y 坐标（基线对齐）。通过聚类算法将零散的文字块合并为完整的文本行。

**聚类策略：**
- 计算每个框的 Y 中心坐标 `(y1 + y2) / 2`
- 按 Y 坐标排序
- 遍历排序后的框，如果相邻框的 Y 差值 ≤ 12 像素，归为同一行
- 每行内的所有框按 X 坐标排序后合并

### 4. 图像提取与保存

为每个合并后的框添加 3 像素边距，裁剪并保存为独立 PNG 图像，文件名格式：

```
box_{全局编号:03d}_{栏}_{局部编号:02d}.png
# 例如：box_001_left_01.png, box_035_right_01.png
```

## 示例输出

### 可视化调试图

`extracted/page_01/debug_visualization.png`：
- 青色竖线：检测到的页面分隔线
- 红色矩形框：检测到的文字行
- 白色数字：全局编号（跨两栏连续编号）

### 提取的文字行示例

**box_002_left_02.png**：
```
【支出】tsɿɦɿ ts'əʔɦɿ ＝〖出賬〗ts'əʔɦɿ
```

**box_041_right_07.png**：
```
【蜘蛛網】tsɿɦɿ tsɿɦɿ mãʟʟ ẑ 【结蛛（羅）網】
```

**box_052_right_18.png**：
```
【豬肝】tsɿɦɿ kəʟʟ 豬的肝
```

## 处理结果统计

前十页处理结果：

| 页码 | 原始连通域 | 左栏行数 | 右栏行数 | 总提取框数 |
|------|-----------|---------|---------|-----------|
| Page 1 | 742 | 34 | 35 | 69 |
| Page 2 | 854 | 37 | 37 | 74 |
| Page 3 | 869 | 36 | 36 | 72 |
| Page 4 | 866 | 35 | 35 | 70 |
| Page 5 | 862 | 36 | 35 | 71 |
| Page 6 | 877 | 37 | 34 | 71 |
| Page 7 | 875 | 39 | 36 | 75 |
| Page 8 | 874 | 35 | 37 | 72 |
| Page 9 | 881 | 36 | 36 | 72 |
| Page 10 | 368 | 17 | 15 | 32 |
| **总计** | **7268** | **342** | **336** | **678** |

## 注意事项

1. **PDF 提取质量**：使用 2x 缩放（`Matrix(2, 2)`）提取页面，确保文字清晰
2. **Y 坐标容忍度**：默认 12 像素，适用于标准排版。如果行间距很小或很大，可能需要调整
3. **页面布局**：当前算法假设页面为标准的左右双栏布局，中间有竖线分隔
4. **第 10 页异常**：第 10 页是章节末尾（"zɿ" 音结尾），内容较少，因此提取框数较少

## 扩展与改进

可能的改进方向：

- **OCR 集成**：结合 Tesseract 或 EasyOCR 直接输出文字内容
- **IPA 音标单独提取**：识别并单独保存 IPA 国际音标部分
- **表格检测**：检测并处理词典中的表格区域
- **多线程处理**：加速批量处理速度

## License

MIT License - 仅供学术研究使用

## 致谢

- 《上海方言词典》原作者和出版方
- OpenCV 和 PyMuPDF 开发团队
