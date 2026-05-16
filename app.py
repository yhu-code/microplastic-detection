from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import uuid
app = Flask(__name__)

def get_peak_pixel(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    lines = gray[h//2 - 10 : h//2 + 10, :]
    line = np.mean(lines, axis=0)

    peak_index = np.argmax(line)

    return peak_index, w


def calibrate():
    green_x, w = get_peak_pixel("green.png")
    red_x, _ = get_peak_pixel("red.png")

    lambda_green = 550
    lambda_red = 580

    a = (lambda_red - lambda_green) / (red_x - green_x)
    b = lambda_green - a * green_x

    return a, b
def pixel_to_wavelength(x, a, b):
    return a * x + b
# ✅ 解决 Not Found（主页）
@app.route('/')
def home():
    return render_template("index.html")


# ✅ 图像处理接口
@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['file']

    # =========================
    # 生成唯一文件名
    # =========================

    filename = str(uuid.uuid4()) + ".png"

    # 保存路径
    filepath = os.path.join("static", filename)

    # 保存上传图片
    file.save(filepath)

    # 读取图片
    img = cv2.imread(filepath)

    if img is None:
        return jsonify({"result": "图片读取失败", "R": 0})

    # 转灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 取中间一行（光谱线）
    h, w = gray.shape
    # 取中间上下10行做平均（更稳定）
    lines = gray[max(0, h//2 - 10) : min(h, h//2 + 10), :]
    line = np.mean(lines, axis=0)

    # 标定
    a, b = calibrate()

    # 生成波长
    wavelengths = [pixel_to_wavelength(i, a, b) for i in range(len(line))]
    # 画图
    plt.figure()
    plt.plot(wavelengths, line)
    # 👉 找峰值
    max_index = np.argmax(line)
    max_wl = wavelengths[max_index]

    # 👉 标记峰
    plt.scatter(max_wl, line[max_index], color='black')
    plt.text(max_wl, line[max_index], f"{int(max_wl)}nm")

    # 标记 550 和 580
    plt.axvline(x=550, color='r', linestyle='--', label='550nm')
    plt.axvline(x=580, color='g', linestyle='--', label='580nm')

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.title("Spectrum")
    plt.legend()

    # 保存到 static 文件夹
    spectrum_name = str(uuid.uuid4()) + ".png"

    spectrum_path = os.path.join("static", spectrum_name)

    plt.savefig(spectrum_path)
    plt.close()
    # ✅ 像素 → 波长（400nm ~ 700nm）

    # =========================
    # 计算 550nm 与 580nm 波段面积
    # =========================

    I550 = 0
    I580 = 0
    A580 = 0

    for i in range(len(line)):

        wl = pixel_to_wavelength(i, a, b)

        intensity = line[i]

        # 540~560nm
        if 540 <= wl <= 560:
            I550 += intensity

        # 570~590nm
        if 570 <= wl <= 590:
            I580 += intensity

        # 560~600nm
        if 560 <= wl <= 600:
            A580 += intensity

    # =========================
    # 计算 R 值
    # =========================

    R = I550 / (I580 + 1)

    # =========================
    # 微塑料干扰判断
    # =========================

    if R < 0.5:
        result = "干扰较小，结果可靠"

    elif R < 1:
        result = "存在部分干扰"

    else:
        result = "干扰较大"

    # =========================
    # 浓度估计模型
    # C = k*A580 + b
    # =========================

    k = 0.045
    b = 2.1

    concentration = k * A580 + b

    # =========================
    # 返回结果
    # =========================

    return jsonify({

        "result": result,

        "R": float(R),

        "concentration": round(concentration, 2),
        "spectrum": spectrum_name
    })
# 启动程序
if __name__ == "__main__":
    app.run(debug=True)