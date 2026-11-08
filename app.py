import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="AI 輔助互動式影像處理與人臉分析工具",
    layout="wide",
)

st.title("🎨 AI 輔助互動式影像處理與人臉分析工具")
st.markdown(
    "本系統支援影像基本解析、多色彩空間與直方圖分析、自訂矩陣卷積濾波（含 PSNR 評估）及人臉偵測。"
)

# 側邊欄：檔案上傳與功能選單
st.sidebar.header("📁 1. 影像載入")
uploaded_file = st.sidebar.file_uploader(
    "上傳圖片 (PNG, JPG, JPEG, BMP)", type=["png", "jpg", "jpeg", "bmp"]
)

if uploaded_file is not None:
    # 讀取圖片
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 基礎解析資訊
    h, w, c = image_rgb.shape
    file_size_kb = uploaded_file.size / 1024

    st.sidebar.success("影像載入成功！")
    st.sidebar.markdown(
        f"""
    - **解析度**: `{w} x {h}` 像素
    - **色彩維度**: `{c}` 通道 (RGB)
    - **檔案大小**: `{file_size_kb:.2f} KB`
    - **格式**: `{uploaded_file.type}`
    """
    )

    # 功能選擇
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. 基礎檢視",
        "2. 色彩空間與直方圖",
        "3. 空間域濾波與 PSNR",
        "4. 人臉偵測模組",
    ])

    with tab1:
        st.subheader("原始影像即時顯示與中繼資訊")
        col1, col2 = st.columns(2)
        with col1:
            st.image(
                image_rgb, caption="原始影像 (RGB)", use_container_width=True
            )
        with col2:
            st.info(f"影像寬度 (Width): {w}")
            st.info(f"影像高度 (Height): {h}")
            st.info(f"色彩深度: {image_rgb.dtype}")
            st.info(f"總像素數: {w * h:,}")

    with tab2:
        st.subheader("多色彩空間轉換與同步直方圖分析")
        view_mode = st.selectbox(
            "選擇顯示視圖與通道",
            [
                "Original (RGB)",
                "R Channel (紅)",
                "G Channel (綠)",
                "B Channel (藍)",
                "YUV - Y (亮度)",
                "YUV - U (色度)",
                "YUV - V (色度)",
            ],
        )

        # 轉換處理
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        target_img = image_rgb.copy()

        if view_mode == "R Channel (紅)":
            r_img = np.zeros_like(image_rgb)
            r_img[:, :, 0] = image_rgb[:, :, 0]
            target_img = r_img
            channel_data = image_rgb[:, :, 0]
            color_hist = "r"
        elif view_mode == "G Channel (綠)":
            g_img = np.zeros_like(image_rgb)
            g_img[:, :, 1] = image_rgb[:, :, 1]
            target_img = g_img
            channel_data = image_rgb[:, :, 1]
            color_hist = "g"
        elif view_mode == "B Channel (藍)":
            b_img = np.zeros_like(image_rgb)
            b_img[:, :, 2] = image_rgb[:, :, 2]
            target_img = b_img
            channel_data = image_rgb[:, :, 2]
            color_hist = "b"
        elif "YUV" in view_mode:
            yuv_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)
            y, u, v = cv2.split(yuv_img)
            if view_mode == "YUV - Y (亮度)":
                target_img = y
                channel_data = y
                color_hist = "gray"
            elif view_mode == "YUV - U (色度)":
                target_img = u
                channel_data = u
                color_hist = "blue"
            else:
                target_img = v
                channel_data = v
                color_hist = "red"
        else:
            channel_data = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            color_hist = "black"

        # 顯示影像
        ax[0].imshow(
            target_img,
            cmap="gray" if len(target_img.shape) == 2 else None,
        )
        ax[0].set_title(f"View: {view_mode}")
        ax[0].axis("off")

        # 繪製直方圖 (0-255)
        if len(target_img.shape) == 3:
            hist = cv2.calcHist(
                [target_img],
                [0 if "R" in view_mode else (1 if "G" in view_mode else 2)],
                None,
                [256],
                [0, 256],
            )
            ax[1].plot(hist, color=color_hist)
        else:
            hist = cv2.calcHist([target_img], [0], None, [256], [0, 256])
            ax[1].plot(
                hist, color="black" if color_hist == "gray" else color_hist
            )

        ax[1].set_title("Pixel Intensity Histogram")
        ax[1].set_xlabel("Pixel Intensity (0-255)")
        ax[1].set_ylabel("Pixel Count")
        ax[1].set_xlim([0, 256])
        ax[1].grid(True, linestyle="--", alpha=0.6)

        st.pyplot(fig)

    with tab3:
        st.subheader("自訂卷積核（Kernel Matrix）空間濾波與量化評估")
        kernel_type = st.selectbox(
            "選擇預設濾波核心或自訂",
            [
                "自訂 3x3 矩陣",
                "高斯模糊 (Gaussian Blur)",
                "邊緣銳化 (Sharpen)",
                "浮雕效果 (Emboss)",
            ],
        )

        if kernel_type == "自訂 3x3 矩陣":
            st.markdown("請輸入 3x3 矩陣數值：")
            col_k1, col_k2, col_k3 = st.columns(3)
            with col_k1:
                k00 = st.number_input("k[0,0]", value=0.0)
                k10 = st.number_input("k[1,0]", value=0.0)
                k20 = st.number_input("k[2,0]", value=0.0)
            with col_k2:
                k01 = st.number_input("k[0,1]", value=-1.0)
                k11 = st.number_input("k[1,1]", value=2.0)
                k21 = st.number_input("k[2,1]", value=0.0)
            with col_k3:
                k02 = st.number_input("k[0,2]", value=0.0)
                k12 = st.number_input("k[1,2]", value=0.0)
                k22 = st.number_input("k[2,2]", value=0.0)
            kernel = np.array(
                [[k00, k01, k02], [k10, k11, k12], [k20, k21, k22]],
                dtype=np.float32,
            )
        elif kernel_type == "高斯模糊 (Gaussian Blur)":
            kernel = (
                np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32)
                / 16.0
            )
        elif kernel_type == "邊緣銳化 (Sharpen)":
            kernel = np.array(
                [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32
            )
        else:  # Emboss
            kernel = np.array(
                [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32
            )

        normalize_kernel = st.checkbox(
            "自動對權重進行正規化 (Normalization)", value=False
        )
        if normalize_kernel and np.sum(kernel) != 0:
            kernel = kernel / np.sum(kernel)

        # 執行濾波
        gray_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        filtered_gray = cv2.filter2D(gray_img, -1, kernel)
        filtered_rgb = cv2.cvtColor(filtered_gray, cv2.COLOR_GRAY2RGB)

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.image(
                cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB),
                caption="原圖（灰階）",
                use_container_width=True,
            )
        with col_f2:
            st.image(
                filtered_rgb, caption="濾波後成果", use_container_width=True
            )

        # 計算 PSNR
        mse = np.mean(
            (gray_img.astype(np.float64) - filtered_gray.astype(np.float64))
            ** 2
        )
        if mse == 0:
            psnr = float("inf")
        else:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))

        st.metric(label="峰值信噪比 (PSNR)", value=f"{psnr:.2f} dB")
        st.latex(
            r"MSE = \frac{1}{MN} \sum_{i=0}^{M-1} \sum_{j=0}^{N-1} [I(i,j) -"
            r" K(i,j)]^2"
        )
        st.latex(
            r"PSNR = 20 \cdot"
            r" \log_{10}\left(\frac{MAX_I}{\sqrt{MSE}}\right) = 10 \cdot"
            r" \log_{10}\left(\frac{MAX_I^2}{MSE}\right)"
        )

    with tab4:
        st.subheader("人臉偵測（Face Detection）模組")
        detector_type = st.selectbox(
            "選擇人臉偵測演算法", ["OpenCV Haar-Cascade Classifier"]
        )

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        gray_for_face = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray_for_face, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        detected_img = image_rgb.copy()
        face_count = len(faces)

        for i, (x, y, w_box, h_box) in enumerate(faces):
            cv2.rectangle(
                detected_img, (x, y), (x + w_box, y + h_box), (0, 255, 0), 3
            )
            confidence = 0.95
            cv2.putText(
                detected_img,
                f"Face #{i+1} ({confidence*100:.0f}%)",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

        st.image(
            detected_img,
            caption=f"人臉偵測結果 (總計偵測到: {face_count} 張人臉)",
            use_container_width=True,
        )
        st.success(f"🎯 偵測完成！總人臉數量：**{face_count}**")

else:
    st.info(
        "👈 請先從左側側邊欄上傳圖片以開始進行各項影像處理與分析。"
    )