import os
import cv2
import numpy as np
import argparse


# 이미지에 전체적으로 따뜻한 색감을 더하는 함수
def warm_tone(img_bgr):
    # 계산 편의를 위해 float32 타입으로 변환
    img = img_bgr.astype(np.float32)

    # OpenCV는 BGR 순서를 사용하므로 채널 분리
    b, g, r = cv2.split(img)

    # 각 채널에 가중치를 곱하여 따뜻한 톤으로 조정
    r = np.clip(r * 1.08, 0, 255)
    g = np.clip(g * 1.03, 0, 255)
    b = np.clip(b * 0.95, 0, 255)

    # 다시 BGR 이미지로 합친 후 uint8로 변환
    warmed = cv2.merge([b, g, r]).astype(np.uint8)
    return warmed


# 채도와 밝기를 약간 높여 색을 더 생동감 있게 만드는 함수
def adjust_saturation(img_bgr, sat_scale=1.12, val_scale=1.04):
    # BGR 이미지를 HSV 색공간으로 변환
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    # H(색상), S(채도), V(명도) 채널 분리
    h, s, v = cv2.split(hsv)

    # 채도와 밝기에 각각 스케일을 곱하여 조정
    s = np.clip(s * sat_scale, 0, 255)
    v = np.clip(v * val_scale, 0, 255)

    # 다시 HSV로 합치고 uint8로 변환
    hsv = cv2.merge([h, s, v]).astype(np.uint8)

    # 최종적으로 다시 BGR 이미지로 변환하여 반환
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# K-means를 이용하여 이미지 색상을 단순화하는 함수
def color_quantization(img_bgr, k=12):
    # 이미지를 (픽셀 수, 3채널) 형태로 펼침
    data = img_bgr.reshape((-1, 3)).astype(np.float32)

    # K-means 종료 조건 설정
    # 최대 30번 반복하거나, 중심 변화가 1.0 이하가 되면 종료
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        1.0
    )

    # K-means 클러스터링 수행
    _, labels, centers = cv2.kmeans(
        data,
        k,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    # 중심값을 정수형으로 변환
    centers = np.uint8(centers)

    # 각 픽셀을 자신이 속한 중심 색상으로 대체
    quantized = centers[labels.flatten()]

    # 원래 이미지 크기로 복원
    quantized = quantized.reshape(img_bgr.shape)

    return quantized


# 부드러운 외곽선을 추출하는 함수
def soft_edges(img_bgr):
    # 외곽선 추출을 위해 그레이스케일로 변환
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 잡음을 줄이기 위해 median blur 적용
    gray = cv2.medianBlur(gray, 5)

    # adaptive threshold를 사용하여 외곽선 마스크 생성
    edges = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        9,
        7
    )

    # 너무 강하고 작은 잡선들을 줄이기 위해 opening 연산 수행
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

    return edges


# bilateral filter를 여러 번 적용하여 이미지를 부드럽게 만드는 함수
def smooth_image(img_bgr, repeat=2):
    # 원본 보존을 위해 복사본 생성
    out = img_bgr.copy()

    # 지정한 횟수만큼 반복해서 bilateral filter 적용
    for _ in range(repeat):
        out = cv2.bilateralFilter(out, d=9, sigmaColor=60, sigmaSpace=60)

    return out


# 전체 지브리풍 렌더링 과정을 수행하는 핵심 함수
def ghibli_style_render(img_bgr, k=12):
    # 이미지 내부를 부드럽게 만들어 디테일을 완화
    smooth = smooth_image(img_bgr, repeat=2)

    # 색상을 단순화하여 평면적인 느낌 강화
    quantized = color_quantization(smooth, k=k)

    # 채도와 밝기를 조금 높여 색을 더 선명하게 만듦
    vivid = adjust_saturation(quantized, sat_scale=1.10, val_scale=1.03)

    # 따뜻한 톤을 추가하여 지브리풍 분위기 강화
    warm = warm_tone(vivid)

    # 원본 이미지 기준으로 부드러운 외곽선 생성
    edges = soft_edges(img_bgr)

    # 외곽선 마스크를 이용해 최종 만화 스타일 이미지 생성
    result = cv2.bitwise_and(warm, warm, mask=edges)

    # 결과 이미지와 중간 과정 결과들을 함께 반환
    return result, edges, quantized, warm


# 비교용 이미지를 저장하는 함수
def save_comparison(original, quantized, warm, edges, result, out_path):
    # 외곽선 이미지는 흑백이므로 시각화를 위해 BGR로 변환
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # 위쪽: 원본 이미지 | 색상 단순화 이미지
    top = np.hstack([original, quantized])

    # 아래쪽: 따뜻한 톤 이미지 | 최종 결과 이미지
    bottom = np.hstack([warm, result])

    # 위아래로 이어붙여 하나의 비교 이미지 생성
    compare = np.vstack([top, bottom])

    # 비교 이미지 저장
    cv2.imwrite(out_path, compare)

def main():
    # 명령줄 인자 파서 생성
    parser = argparse.ArgumentParser(description="Ghibli-like cartoon rendering with OpenCV")

    # 입력 이미지 경로 (필수)
    parser.add_argument("--input", required=True, help="Input image path")

    # 출력 이미지 경로 (기본값 제공)
    parser.add_argument("--output", default="output/result.png", help="Output image path")

    # 색상 개수 k값 (기본값 12)
    parser.add_argument("--k", type=int, default=12, help="Number of colors")

    # 인자 파싱
    args = parser.parse_args()

    # 입력 이미지 읽기
    img = cv2.imread(args.input)

    # 이미지 로드 실패 시 예외 발생
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {args.input}")

    # 지브리풍 렌더링 수행
    result, edges, quantized, warm = ghibli_style_render(img, k=args.k)

    # 출력 폴더가 없으면 자동 생성
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 최종 결과 이미지 저장
    cv2.imwrite(args.output, result)

    # 비교용 이미지 파일 경로 생성
    compare_path = os.path.splitext(args.output)[0] + "_compare.png"

    # 비교 이미지 저장
    save_comparison(img, quantized, warm, edges, result, compare_path)

    # 저장 결과 출력
    print(f"Saved result: {args.output}")
    print(f"Saved comparison: {compare_path}")

if __name__ == "__main__":
    main()