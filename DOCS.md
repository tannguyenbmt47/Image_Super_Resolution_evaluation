# Tài liệu dự án — Siêu phân giải ảnh (Super-Resolution)

Tài liệu dành cho người mới vào dự án. Đọc xong bạn hiểu dự án làm gì, được tổ chức
thế nào, và muốn thay đổi gì thì sửa ở đâu.

---

## 1. Dự án này làm gì

Dự án tăng độ phân giải của ảnh. Đầu vào là một ảnh low-resolution (viết tắt LR),
ít điểm ảnh và mờ. Đầu ra là ảnh high-resolution (viết tắt HR) lớn gấp 4 lần, nhiều
điểm ảnh hơn và rõ hơn. Ví dụ ảnh LR 64×64 điểm ảnh được tạo thành ảnh HR 256×256
điểm ảnh.

Khi một ảnh bị giảm độ phân giải (downsampling), phần lớn chi tiết nhỏ bị mất và
không thể lấy lại bằng phép phóng to thông thường. Model phải tái tạo lại các chi
tiết đã mất, dựa trên quy luật mà nó học được từ rất nhiều ảnh trong quá trình
training.

Dự án cài đặt nhiều model khác nhau, chạy chúng trên cùng một quy trình, rồi đo
chất lượng ảnh đầu ra bằng các metric để so sánh xem model nào tốt hơn.

---

## 2. Cách tổ chức dự án

Dự án dựa trên hai cơ chế. Hiểu hai cơ chế này là hiểu được phần lớn cách vận hành.

### 2.1. Config điều khiển toàn bộ

Mỗi thí nghiệm được mô tả đầy đủ trong một file config dạng YAML, đặt trong thư mục
`configs/`. File này ghi rõ mọi lựa chọn của thí nghiệm. Chương trình chỉ đọc file
config rồi làm đúng theo. Khi muốn đổi thí nghiệm, bạn sửa file config chứ không sửa
mã nguồn.

Ví dụ thật, file `configs/srcnn_df2k_x4.yaml`:

```yaml
model:
  name: SRCNN              # dùng model tên SRCNN
  args: { scale: 4 }       # phóng to 4 lần

train:
  epochs: 200              # số vòng huấn luyện
  batch_size: 16           # số ảnh xử lý mỗi lượt
  loss: charbonnier        # cách đo sai lệch khi học
  lr: 1.0e-4               # learning rate, tốc độ điều chỉnh model
  val_every: 10            # cứ 10 epoch thì đánh giá một lần

train_dataset:
  name: DIV2K              # dùng loader tên DIV2K để nạp ảnh train
  args:
    hr_dir:                # các thư mục ảnh HR (gộp lại thành DF2K)
      - data/DIV2K/DIV2K_train_HR
      - data/Flickr2K/Flickr2K_HR
    scale: 4
    patch_size: 128        # cắt mỗi ảnh thành mảnh 128×128 để học
    pre_upscale: true      # phóng to LR sẵn về cỡ HR (SRCNN cần điều này)
    degradation: bicubic   # tạo LR bằng cách thu nhỏ bicubic

metrics:                   # các metric dùng khi đánh giá
  - { name: psnr,  args: { crop: 4, y_channel: true } }
  - { name: ssim,  args: { crop: 4, y_channel: true } }
  - { name: lpips, args: { net: alex } }
  - niqe
  - musiq
  - clipiqa
```

Muốn train lâu hơn thì sửa `epochs`. Muốn đổi sang model khác thì sửa `name` trong
mục `model`. Không cần đụng tới code.

### 2.2. Registry: cơ chế đăng ký tên

Mỗi model, mỗi dataset, mỗi metric nằm trong một file riêng và tự khai báo tên của
nó vào một danh sách chung gọi là registry. Khi file config ghi một cái tên, chương
trình tra trong registry để tìm đúng thành phần mang tên đó.

Ví dụ thật, file `src/models/srcnn.py` đăng ký model SRCNN:

```python
from . import MODELS

@MODELS.register("SRCNN")          # dòng đăng ký tên SRCNN vào registry
class SRCNN(nn.Module):
    def __init__(self, num_channels=3, scale=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_channels, kernel_size=5, padding=2),
        )

    def forward(self, x):           # nhận ảnh x, trả về ảnh đã làm nét
        return self.features(x)
```

Nhờ dòng `@MODELS.register("SRCNN")`, khi config ghi `name: SRCNN` thì chương trình
biết phải dùng đúng class này. Muốn thêm một model mới, bạn chỉ cần tạo một file
tương tự và đăng ký một tên mới. Không phải sửa chỗ nào khác.

---

## 3. Ba thành phần của một thí nghiệm

Một thí nghiệm luôn gồm model, dataset và metric.

### 3.1. Model

Model là mạng neural network nhận ảnh LR và tạo ra ảnh HR. Dự án có hai model theo
hai hướng tiếp cận khác nhau.

SRCNN là một convolutional neural network (CNN). Nó tạo ảnh HR trong một lần xử lý:
đưa ảnh LR vào, nhận ảnh HR ra ngay. Cấu trúc gồm ba lớp convolution, đơn giản, nhẹ
và chạy nhanh. Đây là model cơ bản công bố năm 2014. File: `src/models/srcnn.py`.

SR3 là một diffusion model. Nó tạo ảnh theo nhiều bước. Bắt đầu từ một ảnh chỉ gồm
nhiễu ngẫu nhiên, qua hàng chục đến hàng trăm bước lặp, mỗi bước loại bớt một phần
nhiễu, ảnh dần hiện rõ thành ảnh HR. Ảnh LR đầu vào được đưa vào mỗi bước để dẫn
hướng cho quá trình, đảm bảo ảnh tạo ra đúng với ảnh gốc. SR3 cho ảnh chi tiết và tự
nhiên hơn SRCNN, nhưng chậm hơn nhiều vì phải lặp qua nhiều bước. File:
`src/models/sr3.py`, phần lõi diffusion ở `src/models/diffusion/`.

### 3.2. Dataset

DF2K là tập ảnh HR dùng để train, gồm khoảng 3450 ảnh ghép từ hai bộ DIV2K và
Flickr2K. Dự án chỉ cần ảnh HR, vì ảnh LR được tạo tự động bằng cách downsampling
ảnh HR. Trong config, DF2K được biểu diễn bằng một danh sách thư mục, như đoạn
`hr_dir` ở mục 2.1.

DIV2K-Val là 100 ảnh được tách riêng để đánh giá. Tập này không bao giờ được dùng để
train, để điểm số phản ánh đúng khả năng của model trên ảnh nó chưa từng thấy.

RealSR là các cặp ảnh LR và HR chụp trực tiếp bằng máy ảnh thật, không phải tạo bằng
downsampling. Tập này dùng cho training giai đoạn 2 và đánh giá trên ảnh thực tế.
File loader: `src/datasets/realsr.py`.

Ảnh LR được tạo theo hai kiểu degradation, đặt trong config bằng trường
`degradation`. Kiểu `bicubic` chỉ thu nhỏ ảnh đơn giản. Kiểu `realistic` mô phỏng
ảnh đời thực bằng cách thêm blur, nhiễu và nén JPEG.

### 3.3. Metric

Sau khi model tạo ra ảnh, cần đo xem ảnh tốt đến đâu. Có hai nhóm metric.

Nhóm full-reference cần ảnh HR gốc để so sánh. PSNR và SSIM đo mức sai khác về điểm
ảnh và về cấu trúc giữa ảnh tạo ra và ảnh gốc; giá trị càng cao càng giống ảnh gốc.
LPIPS cũng so với ảnh gốc nhưng dựa trên đặc trưng trích từ một mạng neural network,
gần với cảm nhận của mắt người hơn; giá trị càng thấp càng tốt.

Nhóm no-reference không cần ảnh gốc. NIQE, MUSIQ và CLIPIQA chấm điểm chỉ dựa trên
bản thân ảnh tạo ra, đánh giá ảnh có sắc nét và tự nhiên hay không. Nhóm này hữu ích
vì điểm full-reference đôi khi không phản ánh đúng mức độ đẹp mà mắt người cảm nhận.

---

## 4. Quy trình làm việc

### 4.1. Training

Chương trình đưa cho model rất nhiều cặp ảnh LR và HR. Với mỗi cặp, model dự đoán
ảnh HR từ ảnh LR, chương trình so kết quả với ảnh HR thật để tính loss, là con số đo
mức sai lệch, rồi điều chỉnh các tham số bên trong model để lần sau loss nhỏ hơn.
Quá trình này lặp lại hàng nghìn lần, model dần chính xác hơn.

Cứ sau một số epoch (đặt bằng `val_every` trong config), chương trình tự chạy model
trên DIV2K-Val và in điểm ra màn hình. File model đã lưu gọi là checkpoint, đặt
trong thư mục `experiments/`. Chương trình lưu hai checkpoint: `best.pth` được lưu
lại mỗi khi điểm validation tốt lên, và `last.pth` lưu ở lần cuối. Khi đánh giá nên
dùng `best.pth`.

Để tránh train thừa, chương trình có early stopping. Nó theo dõi một metric
(mặc định là psnr, đặt bằng `early_stop_metric`). Nếu sau một số lần validation liên
tiếp mà metric không tốt lên (số lần đặt bằng `early_stop_patience`), chương trình
dừng training sớm, không chạy hết số epoch tối đa. Nhờ vậy có thể đặt `epochs` cao
mà không sợ lãng phí: model hội tụ tới đâu thì dừng tới đó.

### 4.2. Đánh giá (evaluate)

Chương trình nạp một checkpoint, chạy model trên tập test với toàn bộ metric, rồi in
ra một bảng kết quả. Định dạng bảng như sau, mỗi dòng là một tập dữ liệu, mỗi cột là
một metric:

```
dataset             psnr      ssim     lpips      niqe     musiq   clipiqa
------------------------------------------------------------------------
DIV2K-Val          27.31    0.7912    0.241      4.85     58.2     0.62
```

Chạy evaluate cho cả hai model rồi đặt hai bảng cạnh nhau sẽ thấy model nào tốt hơn
ở từng metric.

### 4.3. Hai giai đoạn training

Giai đoạn 1 gọi là pretrain, train model trên DF2K với ảnh LR tạo bằng downsampling.
Sau giai đoạn này model biết làm nét ảnh nói chung.

Giai đoạn 2 gọi là fine-tune, train tiếp model đó trên RealSR với ảnh chụp thật. Cần
giai đoạn này vì ảnh LR tạo bằng downsampling không giống ảnh LR ngoài đời thực. Ảnh
thật còn có blur ống kính, nhiễu cảm biến và mất mát do nén. Fine-tune trên ảnh thật
giúp model chạy tốt với ảnh thật. Khi chạy giai đoạn 2, ta nạp checkpoint của giai
đoạn 1 bằng tham số `--pretrained`.

---

## 5. Cấu trúc thư mục

```
configs/    Các file config. Mỗi file là một thí nghiệm. Bạn sửa ở đây nhiều nhất.
              srcnn_df2k_x4.yaml   train SRCNN trên DF2K
              sr3_df2k_x4.yaml     train SR3 trên DF2K

scripts/    Các lệnh để chạy:
              train.py          train một model
              evaluate.py       đánh giá một checkpoint
              download_data.sh  tải dữ liệu về máy

src/        Phần lõi của chương trình:
              models/    các model, ví dụ srcnn.py, sr3.py
              datasets/  nạp ảnh và tạo ảnh LR từ ảnh HR
              metrics/   các metric, ví dụ psnr.py, ssim.py
              losses.py  cách tính loss khi train model SRCNN
              engine/    vòng lặp training và vòng lặp evaluate
              utils/     phần nền: đọc file config, quản lý registry

tests/      Các bài kiểm tra tự động, chạy nhanh để đảm bảo sửa code không làm hỏng
            phần khác.

data/       Dữ liệu ảnh. Không kèm sẵn trong dự án, phải tải về.
experiments/Nơi lưu checkpoint và log của quá trình training.
```

Khi cần thay đổi:

- Đổi thiết lập thí nghiệm như số epoch, model, dataset: sửa file trong `configs/`.
- Thêm một model mới: xem hướng dẫn chi tiết ở mục 8.
- Thêm một metric mới: thêm một file trong `src/metrics/` và đăng ký tên.
- Các thư mục `engine/` và `utils/` là phần khung, hiếm khi cần sửa.

---

## 6. Cách chạy

```bash
# làm một lần: cài thư viện và tải dữ liệu
.venv/bin/pip install -r requirements.txt
bash scripts/download_data.sh

# train model, chạy riêng từng cái
.venv/bin/python scripts/train.py --config configs/srcnn_df2k_x4.yaml
.venv/bin/python scripts/train.py --config configs/sr3_df2k_x4.yaml

# đánh giá một checkpoint, in ra bảng metric (nên dùng best.pth)
.venv/bin/python scripts/evaluate.py \
    --config configs/srcnn_df2k_x4.yaml \
    --checkpoint experiments/srcnn_df2k_x4/best.pth

# chạy thử nhanh vài epoch để kiểm tra, không phải kết quả thật
.venv/bin/python scripts/train.py --config configs/srcnn_df2k_x4.yaml \
    --epochs 2 --repeat 1 --no-val
```

Tên file config theo quy ước model _ dataset _ scale. Ví dụ `sr3_df2k_x4` là model
SR3, dataset DF2K, scale 4.

---

## 7. Những điều cần biết trước

SR3 chạy rất chậm vì nó tạo ảnh qua nhiều bước. Nên train SR3 trên máy có GPU. SRCNN
nhẹ nên chạy nhanh hơn nhiều.

Batch size được chỉnh để dùng khoảng nửa bộ nhớ GPU 8GB. SRCNN dùng batch size 16,
còn SR3 chỉ dùng batch size 3 vì nó tốn nhiều bộ nhớ hơn.

Khi đánh giá, SR3 chỉ chạy trên phần ảnh cắt giữa kích thước 256×256 điểm ảnh, vì
chạy trên ảnh lớn nguyên bản sẽ vượt quá bộ nhớ GPU. SRCNN đánh giá trên ảnh đầy đủ.
Do hai model chạy đánh giá trên kích thước ảnh khác nhau, điểm số của chúng chưa so
sánh trực tiếp được, trừ khi cho cả hai chạy trên cùng một kích thước.

Hai model trong dự án được cài đặt lại theo đúng ý tưởng của paper gốc, nhưng không
phải bản chính thức của tác giả. Vì vậy không nạp được các tham số đã train sẵn mà
tác giả gốc công bố.

Training đầy đủ để đạt kết quả tốt cần nhiều thời gian. Các con số từ lần chạy thử
nhanh chỉ để xác nhận chương trình hoạt động, chưa phải kết quả tốt. Khi train quá
ít epoch, model thậm chí cho ảnh kém hơn phép phóng to bicubic thông thường, vì nó
cần học rất lâu mới đạt chất lượng cao.

---

## 8. Hướng dẫn thêm một model mới

Thêm một model gồm bốn bước. Không phải sửa code ở `engine/` hay các script.

### Bước 1: Tạo file model trong `src/models/`

Tạo file mới, ví dụ `src/models/mynet.py`. Định nghĩa một class kế thừa
`nn.Module` và đăng ký nó vào registry bằng decorator `@MODELS.register("MyNet")`.
Tên trong ngoặc là tên bạn sẽ ghi trong config.

Constructor nhận các tham số lấy từ mục `model.args` trong config. Nên có tham số
`scale` (độ phóng) và `num_channels` (số kênh màu, mặc định 3) để đồng nhất với các
model khác.

Ví dụ một feed-forward model tự phóng to ảnh bằng PixelShuffle:

```python
import torch.nn as nn

from . import MODELS

@MODELS.register("MyNet")
class MyNet(nn.Module):
    def __init__(self, scale=4, num_channels=3, n_feats=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(num_channels, n_feats, 3, padding=1),
            nn.ReLU(inplace=True),
            # tạo đủ kênh rồi PixelShuffle để phóng to scale lần
            nn.Conv2d(n_feats, num_channels * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
        )

    def forward(self, x):
        # x là batch ảnh LR, kích thước [N, C, H, W], giá trị trong [0, 1]
        return self.body(x)   # trả về ảnh SR, kích thước [N, C, H*scale, W*scale]
```

### Bước 2: Khai báo file trong `src/models/__init__.py`

Decorator chỉ chạy khi file được import. Mở `src/models/__init__.py` và thêm một
dòng import cạnh các dòng có sẵn:

```python
from . import srcnn
from . import sr3
from . import mynet   # dòng thêm mới
```

### Bước 3: Tạo config trỏ tới model

Tạo file trong `configs/`, ví dụ `configs/mynet_df2k_x4.yaml`, với mục `model` ghi
đúng tên đã đăng ký. Các giá trị trong `args` chính là tham số của constructor:

```yaml
model:
  name: MyNet
  args: { scale: 4, n_feats: 64 }
```

Lưu ý về `pre_upscale`: model ví dụ trên tự phóng to ảnh bên trong, nên dataset đặt
`pre_upscale: false`. Nếu model của bạn cần ảnh LR đã phóng to sẵn về cỡ HR (như
SRCNN), thì dataset phải đặt `pre_upscale: true`. Model và dataset phải khớp điểm
này nếu không kích thước ảnh sẽ lệch.

### Bước 4: Chạy

```bash
.venv/bin/python scripts/train.py --config configs/mynet_df2k_x4.yaml
```

### Hai loại interface của model

Engine tự nhận biết model thuộc loại nào dựa trên các hàm mà model định nghĩa:

- **Feed-forward**: chỉ cần định nghĩa `forward(self, x)` trả về ảnh SR. Engine sẽ
  train bằng pixel loss (giống SRCNN). Đây là loại đơn giản nhất.
- **Tự định nghĩa cách train và suy luận**: định nghĩa hai hàm
  `compute_loss(self, lr, hr)` trả về một con số loss, và
  `super_resolve(self, lr)` trả về ảnh SR. Engine sẽ dùng hai hàm này thay cho pixel
  loss và `forward`. Diffusion model SR3 thuộc loại này, vì nó train bằng cách dự
  đoán nhiễu và tạo ảnh bằng nhiều bước lặp.

### Quy ước về tensor

Mọi ảnh trong dự án là tensor số thực, giá trị trong khoảng [0, 1], thứ tự kênh đặt
trước là [N, C, H, W] với N là số ảnh trong batch, C là số kênh màu (3 cho ảnh RGB).
Model phải nhận và trả về đúng quy ước này.

### Khuyến nghị: thêm test

Thêm một test hình dạng đầu ra vào `tests/unit/test_models.py` để chắc model chạy
đúng kích thước. Ví dụ, đưa vào ảnh LR 8×8 thì với scale 4 phải ra ảnh 32×32.
