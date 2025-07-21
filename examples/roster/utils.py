import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.colors as mcolors


def total_data2single_data(total_data, span):
    single_data = []
    span = min(span, total_data["seq_len"])
    for i in range(1, span+1):
        item_data = {}
        for k, v in total_data.items():
            if type(v) is list:
                item_data[k] = v[:i]
            else:
                item_data[k] = v
        item_data["seq_len"] = i
        single_data.append(item_data)
    return single_data


def data2batches(data, batch_size):
    batches = []
    batch = []
    for item_data in data:
        if len(batch) < batch_size:
            batch.append(item_data)
        else:
            batches.append(batch)
            batch = [item_data]
    if len(batch) > 0:
        batches.append(batch)
    return batches


def trace_related_cs_change(cs_state_seq, concept_seq, correctness_seq, figsize=(22, 3)):
    cs_state_seq = np.array(cs_state_seq)
    concept_seq = np.array(concept_seq)
    correctness_seq = np.array(correctness_seq)
    
    T, N = cs_state_seq.shape

    related_ks = sorted(set(concept_seq.tolist()))
    ks2row = {ks: i for i, ks in enumerate(related_ks)}

    heatmap_data = np.zeros((len(related_ks), T))
    for i, ks in enumerate(related_ks):
        heatmap_data[i] = cs_state_seq[:, ks]

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(related_ks))]

    fig, ax = plt.subplots(figsize=figsize)

    # 画热力图，去掉矩形框边缘 linewidth=0
    im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlGn', interpolation='none', vmin=0, vmax=1)
    # 获取当前热力图位置
    pos = ax.get_position()

    # 在右边新建一个 axes 用于 colorbar，控制位置
    cbar_ax = fig.add_axes([
        pos.x1 + 0.01,     # x: 右边稍微偏移
        pos.y0 + 0.01,     # y: 从热力图底部偏上开始（往下移）
        0.01,              # width: 窄条
        pos.height * 0.7   # height: 稍微低于原图，避免进圈
    ])

    # 添加 colorbar
    cbar = fig.colorbar(im, cax=cbar_ax)

    for spine in ax.spines.values():
        spine.set_visible(False)
        
    # 左侧实心圆，x向左移一点，避免太靠近y轴
    for i, color in enumerate(colors):
        ax.add_patch(plt.Circle((-1.2, i), 0.3, color=color, transform=ax.transData, clip_on=False))

    # 上方实心或空心圆，空心改成圆环
    ring_radius = 0.3
    for t in range(T):
        ks = concept_seq[t]
        if ks not in ks2row:
            continue
        i = ks2row[ks]
        color = colors[i]
        y = -1

        if correctness_seq[t]:
            # 实心圆
            circle = plt.Circle((t, y), ring_radius, color=color, transform=ax.transData, clip_on=False, zorder=10)
            ax.add_patch(circle)
        else:
            # 圆环：实心色大圆 + 中心白圆覆盖
            hole_radius = ring_radius / 2

            outer_circle = plt.Circle((t, y), ring_radius, color=color, transform=ax.transData, clip_on=False, zorder=10)
            inner_circle = plt.Circle((t, y), hole_radius, color='white', transform=ax.transData, clip_on=False, zorder=11)

            ax.add_patch(outer_circle)
            ax.add_patch(inner_circle)

    ax.set_xlim(-1, T)
    ax.set_ylim(len(related_ks) - 0.5, -1.5)
    ax.set_xlabel("")

    # x轴显示全部时间步刻度
    ax.set_xticks(np.arange(T))
    ax.set_xticklabels(np.arange(T))
    ax.tick_params(axis='x', length=0)  # 设置 x 轴刻度线长度为 0（也可以加 width=0）
    
    ax.set_yticks([])

    # plt.tight_layout()
    # plt.show()

    return fig


def trace_selected_cs_change(cs_state_seq, question_seq, correctness_seq, selected_ks, figsize=(22, 3)):
    cs_state_seq = np.array(cs_state_seq)
    question_seq = np.array(question_seq)
    correctness_seq = np.array(correctness_seq)
    
    T, N = cs_state_seq.shape
    selected_ks = [ks for ks in selected_ks if ks < N]

    heatmap_data = cs_state_seq[:, selected_ks].T  # (len(selected_ks), T)
    num_selected = len(selected_ks)

    fig, ax = plt.subplots(figsize=figsize)

    # 画热力图
    im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlGn', interpolation='none', vmin=0, vmax=1)

    # 设置 ytick 为知识点编号
    ax.set_yticks(np.arange(num_selected))
    ax.set_yticklabels([f"KS {ks}" for ks in selected_ks])

    # 设置 xtick
    ax.set_xticks(np.arange(T))
    ax.set_xticklabels(np.arange(T))
    ax.tick_params(axis='x', labelrotation=0, top=False)

    # 横轴上方题号 & 正误标记
    for t in range(T):
        qid = question_seq[t]
        result = correctness_seq[t]
        mark = '✔' if result else '✘'
        color = 'green' if result else 'red'

        ax.text(t, -1.5, f"q{qid}", ha='center', va='center', fontsize=10)
        ax.text(t, -1, mark, ha='center', va='center', fontsize=12, color=color)

    # 把 y轴左边的文字稍微往右移一点
    ax.tick_params(axis='y', pad=8)

    # 去掉边框
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 添加 colorbar
    pos = ax.get_position()
    cbar_ax = fig.add_axes([
        pos.x1 + 0.01,     # x: 右边稍微偏移
        pos.y0 + 0.01,     # y: 从热力图底部偏上开始（往下移）
        0.01,              # width: 窄条
        pos.height * 0.7   # height: 稍微低于原图，避免进圈
    ])
    cbar = fig.colorbar(im, cax=cbar_ax)

    # 限制 y 轴范围防止文字被裁剪
    ax.set_ylim(num_selected - 0.5, -2)
    
    ax.set_xlabel("")
    ax.set_ylabel("")

    # 设置 ytick 为知识点编号
    ax.set_yticks(np.arange(num_selected))
    ax.set_yticklabels([f"c{ks}" for ks in selected_ks])
    ax.tick_params(axis='y', labelrotation=0, top=False, length=0)

    # 设置 xtick
    ax.set_xticks(np.arange(T))
    ax.set_xticklabels(np.arange(T))
    ax.tick_params(axis='x', labelrotation=0, top=False, length=0)

    # plt.tight_layout()
    # plt.show()

    return fig


def trace_single_concept_change(target_concept, c_state_seq, qc_relation_seq, correctness_seq, figsize=(22, 3)):
    c_state_seq = np.array(c_state_seq)
    qc_relation_seq = np.array(qc_relation_seq)
    correctness_seq = np.array(correctness_seq)
    
    T = len(c_state_seq)
    x = np.arange(T)

    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(0, 1)
    colors = cmap(norm(qc_relation_seq))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, c_state_seq, color='blue', linewidth=2)
    ax.set_title(f"Knowledge Tracing over Time on c{target_concept}")
    
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_xticks([])
    
    ymax = min(1, np.max(c_state_seq) + 0.05)
    ax.set_ylim(0, ymax)  # ✅ y轴只显示 >=0 的部分
    
    ax.tick_params(axis='y', labelrotation=0, top=False, length=0)
    ax.tick_params(axis='x', labelrotation=0, top=False, length=0)

    # ✅ 横轴下方添加标记：用混合坐标
    base_y_axes = -0.2
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    
    def get_color(strength, is_correct):
        """
        给定关联强度和是否做对，返回 RGB 颜色
        strength: 0 ~ 1
        is_correct: True → 绿色，False → 红色
        """
        # HSV: h=0红，h=120绿，s越高越艳，v=1保持亮度
        hue = 120 / 360 if is_correct else 0.0
        saturation = strength
        value = 1.0
        return mcolors.hsv_to_rgb((hue, saturation, value))

    for t in range(T):
        if t > 0:
            # y(t-1) 和 y(t)
            y_prev = c_state_seq[t - 1].item()
            y_curr = c_state_seq[t].item()

            # 虚线1：从 (t-1, y(t-1)) 到 (t, 0)
            ax.plot([t - 1, t - 0.5], [y_prev, 0],
                    linestyle='dashed', color='gray', linewidth=1.0, alpha=0.7)

            # 虚线2：从 (t, 0) 到 (t, y(t))
            ax.plot([t - 0.5, t], [0, y_curr],
                    linestyle='dashed', color='gray', linewidth=1.0, alpha=0.7)
    
            strength = qc_relation_seq[t]
            is_correct = bool(correctness_seq[t])
            color = get_color(strength, is_correct)
            symbol = '✓' if is_correct else '✕'

            # 黑色描边底层
            ax.text(t - 0.5, base_y_axes, symbol,
                    color='black',
                    transform=trans,
                    fontsize=19, ha='center', va='center',
                    clip_on=False, zorder=9)

            # 上层彩色符号
            ax.text(t - 0.5, base_y_axes, symbol,
                    color=color,
                    transform=trans,
                    fontsize=17, ha='center', va='center',
                    clip_on=False, zorder=10)

    # 底部加空间避免裁剪
    plt.subplots_adjust(bottom=0.25)

    # plt.tight_layout()
    # plt.show()

    return fig
