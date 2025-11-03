import itertools
from pathlib import Path

import cv2
import matplotlib.colors
import numpy as np
import pandas as pd
import wfdb.io


class Colors:
    BLACK = (0, 0, 0)
    GRAY = (128, 128, 128)
    LIGHT_GRAY = (224, 224, 224)
    WHITE = (255, 255, 255)
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    CYAN = (0, 255, 255)
    MAGENTA = (255, 0, 255)
    YELLOW = (255, 255, 0)
    BLUE_AZURE = (255, 128, 0)
    GREEN_SPRING = (128, 255, 0)
    GREEN_PARIS = (0, 255, 128)


def write_text(image: cv2.typing.MatLike, text: str | tuple[str, str], pos: int):
    x = 20
    y = 30 + pos * 30
    final_text = f'{text[0]}: {text[1]}' if isinstance(text, tuple) else f'{text}'
    drawn = cv2.putText(image, final_text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (128, 255, 0), 1)
    return drawn


def write_texts(image: cv2.typing.MatLike, texts: list[str]):
    drawn = image.copy()
    for k, text in enumerate(texts):
        drawn = write_text(drawn, text, k)
    return drawn


def main():

    dataset_root = Path('/home/thulio/projects/masters-research/data')

    df_subjects = pd.read_csv('csvs/subjects.csv', index_col=0)

    cv2.namedWindow('plot')
    cv2.moveWindow('plot', 2040 - 215, 0)

    signal_names = ['II', 'PLETH', 'ABP']

    matplotlib_default_colors = matplotlib.colors.to_rgba_array([f'C{i}' for i in range(3)])
    signal_colormap = dict(enumerate((matplotlib_default_colors[:, :-1] * 255).round().astype(np.uint8).tolist()))
    line_type = cv2.LINE_AA

    num_plots = 3
    window_size = (1600, 900)
    window_margins = (40, 40)
    window_width, window_height = window_size
    window_margin_x, window_margin_y = window_margins
    graph_width = window_width - 2 * window_margin_x
    graph_height = (window_height - (num_plots + 1) * window_margin_y) // num_plots
    graph_origins = [(window_margin_x, i * graph_height + (i + 1) * window_margin_y) for i in range(num_plots)]
    graph_default_range = (-1.0, 1.0)

    window = 5 * 60

    subject_idx = 0
    sample_idx = 0

    def draw_graph(frame,
                   graph_origin: tuple[int, int],
                   graph_range: tuple[tuple[float, float], tuple[float, float]]) -> None:
        (min_x, max_x), (min_y, max_y) = graph_range
        graph_origin_x, graph_origin_y = graph_origin
        graph_tl = (graph_origin_x, graph_origin_y)
        graph_br = (graph_origin_x + graph_width, graph_origin_y + graph_height)
        frame = cv2.rectangle(frame, graph_tl, graph_br, Colors.BLACK, lineType=line_type)
        if not np.isnan(min_x) and not np.isnan(max_x):
            order_mag = 10 ** np.floor(min(np.log10(max_x - min_x), 1))
            vline_dist = order_mag / 2 if (max_x - min_x) / (order_mag / 2) < 10 else order_mag
            lower_vline_x = np.ceil(min_x / vline_dist) * vline_dist
            upper_vline_x = np.ceil(max_x / vline_dist) * vline_dist
            for x in np.arange(lower_vline_x, upper_vline_x, vline_dist):
                vline_x_n = (x - min_x) / (max_x - min_x)
                vline_x_p = int(vline_x_n * graph_width + graph_origin_x)
                vline_p_0 = (vline_x_p, graph_origin_y)
                vline_p_1 = (vline_x_p, graph_origin_y + graph_height)
                frame = cv2.line(frame, vline_p_0, vline_p_1, Colors.LIGHT_GRAY, lineType=line_type)
                (tw, th), _ = cv2.getTextSize(f'{x: .2f}', cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 1)
                tp = (vline_x_p - tw // 2, graph_origin_y + graph_height + th + 5)
                frame = cv2.putText(frame, f'{x: .2f}', tp, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, Colors.LIGHT_GRAY, lineType=line_type)
        if min_x <= 0.0 <= max_x:
            axis_x_n = max_x / (max_x - min_x)
            axis_x_p = int(axis_x_n * graph_width + graph_origin_x)
            axis_point_0 = (axis_x_p, graph_origin_y)
            axis_point_1 = (axis_x_p, graph_origin_y + graph_height)
            frame = cv2.line(frame, axis_point_0, axis_point_1, Colors.BLACK, lineType=line_type)
        if min_y <= 0.0 <= max_y:
            axis_y_n = max_y / (max_y - min_y)
            axis_y_p = int(axis_y_n * graph_height + graph_origin_y)
            axis_point_0 = (graph_origin_x, axis_y_p)
            axis_point_1 = (graph_origin_x + graph_width, axis_y_p)
            frame = cv2.line(frame, axis_point_0, axis_point_1, Colors.BLACK, lineType=line_type)
        pos_min_x = (graph_origin_x - 5, graph_origin_y + graph_height + 15)
        pos_max_x = (graph_origin_x + graph_width - 25, graph_origin_y + graph_height + 15)
        pos_min_y = (graph_origin_x - 40, graph_origin_y + graph_height - 5)
        pos_max_y = (graph_origin_x - 40, graph_origin_y + 15)
        frame = cv2.putText(frame, f'{min_x: .2f}', pos_min_x, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, Colors.BLACK, lineType=line_type)
        frame = cv2.putText(frame, f'{max_x: .2f}', pos_max_x, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, Colors.BLACK, lineType=line_type)
        frame = cv2.putText(frame, f'{min_y: .2f}', pos_min_y, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, Colors.BLACK, lineType=line_type)
        frame = cv2.putText(frame, f'{max_y: .2f}', pos_max_y, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, Colors.BLACK, lineType=line_type)
        return frame

    def draw_signal(frame,
                    signal_x,
                    signal_y,
                    graph_origin: tuple[int, int],
                    graph_range: tuple[tuple[float, float], tuple[float, float]],
                    color: Colors) -> None:
        (min_x, max_x), (min_y, max_y) = graph_range
        data_x, data_y = np.array(signal_x), np.array(signal_y)
        graph_origin_x, graph_origin_y = graph_origin
        graph_cond = (data_x >= min_x) & (data_x <= max_x) & (data_y >= min_y) & (data_y <= max_y)
        data_x_p = (data_x[graph_cond] - min_x) / (max_x - min_x) * graph_width + graph_origin_x
        data_y_p = (data_y[graph_cond] - max_y) / (min_y - max_y) * graph_height + graph_origin_y
        groups = itertools.groupby(np.vstack((data_x_p, data_y_p)).T, lambda k: np.all(np.isfinite(k)))
        for isfinite, group in groups:
            if isfinite:
                data_g = np.vstack(list(group)).astype(int)
                frame = cv2.polylines(frame, [data_g], False, color, lineType=line_type)
        return frame

    def plot_signals(frame,
                     time,
                     signals) -> None:
        for s, graph_origin in enumerate(graph_origins):
            signal = signals[:, s]
            graph_range_x = rx if np.isfinite((rx := (np.min(time), np.max(time)))).all() else graph_default_range
            graph_range_y = ry if np.isfinite((ry := (np.min(signal), np.max(signal)))).all() else graph_default_range
            graph_range = (graph_range_x, graph_range_y)
            frame = draw_graph(frame, graph_origin, graph_range)
            frame = draw_signal(frame, time, signal, graph_origin, graph_range, signal_colormap[s])
        return frame

    while True:

        curr_subject = df_subjects.iloc[subject_idx]

        subject_id, record_start, segment_id, master_id, label = curr_subject

        p_folder = dataset_root / label / f'p{subject_id:06d}'
        signals, fields = wfdb.io.rdsamp(p_folder / segment_id, channel_names=signal_names)

        n_samples = fields['sig_len']
        fs = fields['fs']

        time = np.arange(n_samples) / fs

        sa = sample_idx
        sb = sample_idx + int(np.floor(window * fs))

        wt = time[sa:sb]
        ws = signals[sa:sb, :]

        curr_image = np.full((window_height, window_width, 3), 255, dtype=np.uint8)
        curr_image = plot_signals(curr_image, wt, ws)

        texts = [('subject_idx', subject_idx), ('sample_idx', sample_idx)]
        curr_image = write_texts(curr_image, texts)

        # cv2.moveWindow('plot', 1080, 0)
        cv2.imshow('plot', curr_image)

        key = cv2.waitKey()
        if key == ord('q'):
            break

        if key == ord('a'):
            sample_idx = (sample_idx - 1) % n_samples
        elif key == ord('d'):
            sample_idx = (sample_idx + 1) % n_samples
        elif key == ord('s'):
            subject_idx = (subject_idx - 1) % len(df_subjects)
            sample_idx = 0
        elif key == ord('w'):
            subject_idx = (subject_idx + 1) % len(df_subjects)
            sample_idx = 0

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
