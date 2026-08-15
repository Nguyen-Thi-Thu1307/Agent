import time
from pathlib import Path

import pandas as pd
import streamlit as st

import agents as ag

VI_ACTION = {ag.SUCK: "Hút bụi", ag.LEFT: "Qua trái", ag.RIGHT: "Qua phải", ag.NOOP: "NoOp"}
VI_STATUS = {ag.CLEAN: "sạch", ag.DIRTY: "bẩn", ag.UNKNOWN: "chưa biết"}
VI_NAME = ["1 · Bảng điều khiển", "2 · Phản xạ đơn giản", "3 · Phản xạ có trạng thái",
           "4 · Hướng mục đích", "5 · Hướng lợi ích", "6 · Có khả năng học",
           "7 · Học tăng cường"]
Q_ACTION = ["Hút bụi", "Chuyển ô", "NoOp"]

SPEC = [
    dict(en="table-driven agent",
         sense="vị trí hiện tại và trạng thái sạch/bẩn của ô đó",
         memory="toàn bộ chuỗi cảm nhận đã nhận từ đầu",
         rules=["bảng ánh xạ chuỗi cảm nhận → hành động, tra theo hậu tố dài nhất có trong bảng",
                "[A, bẩn] → Hút bụi&nbsp;&nbsp;&nbsp;&nbsp;[B, bẩn] → Hút bụi",
                "[A, sạch] → Qua phải&nbsp;&nbsp;&nbsp;&nbsp;[B, sạch] → Qua trái",
                "[A, sạch][A, sạch] → Qua phải",
                "[A, sạch][A, bẩn] → Hút bụi"],
         note="sáu mục ghi sẵn; không có mục nào khớp thì trả về NoOp",
         limit="Bảng phải liệt kê trước mọi chuỗi cảm nhận có thể xảy ra. Kích thước bảng tăng "
               "theo cấp số nhân với độ dài chuỗi, nên phương pháp chỉ khả thi với bài toán rất nhỏ."),
    dict(en="simple reflex agent",
         sense="vị trí hiện tại và trạng thái sạch/bẩn của ô đó",
         memory="không có",
         rules=["tập luật điều kiện – hành động, so khớp trên cảm nhận hiện tại",
                "[bẩn] → Hút bụi", "[sạch] → di chuyển sang ô còn lại"],
         note="quyết định chỉ phụ thuộc cảm nhận hiện tại, không phụ thuộc chuỗi cảm nhận",
         limit="Không lưu trạng thái bên trong nên không xác định được ô còn lại đã sạch. "
               "Hành động NoOp không bao giờ được chọn; trong môi trường tĩnh, tác tử chịu "
               "chi phí di chuyển ở mọi bước sau khi cả hai ô đã sạch."),
    dict(en="model-based reflex agent",
         sense="vị trí hiện tại và trạng thái sạch/bẩn của ô đó",
         memory="trạng thái sạch/bẩn đã biết của từng ô, cập nhật sau mỗi cảm nhận",
         rules=["cập nhật trạng thái bên trong từ cảm nhận, sau đó so khớp luật",
                "[bẩn] → Hút bụi", "cả hai ô đã biết là sạch → NoOp",
                "ngược lại → di chuyển sang ô còn lại"],
         note="trạng thái bên trong bù phần môi trường mà cảm biến không quan sát được",
         limit="Mục đích nằm ẩn trong tập luật. Thay đổi mục đích đòi hỏi viết lại luật "
               "chứ không chỉ sửa dữ liệu."),
    dict(en="goal-based agent",
         sense="vị trí hiện tại và trạng thái sạch/bẩn của ô đó",
         memory="trạng thái đã biết của hai ô; mục đích biểu diễn tách rời khỏi tập luật",
         rules=["mục đích: cả hai ô ở trạng thái sạch",
                "tìm kiếm theo chiều rộng trên không gian trạng thái để sinh chuỗi hành động đạt mục đích",
                "thực hiện hành động đầu tiên của chuỗi, lập lại kế hoạch ở bước sau",
                "ô chưa quan sát được coi là bẩn — giả định bi quan"],
         note="mục đích là dữ liệu đầu vào của thuật toán tìm kiếm, không phải luật viết cứng",
         limit="Chỉ phân biệt được trạng thái đạt và chưa đạt mục đích. Khi nhiều phương án "
               "cùng đạt mục đích, tác tử không có tiêu chí xếp hạng giữa chúng."),
    dict(en="utility-based agent",
         sense="vị trí hiện tại và trạng thái sạch/bẩn của ô đó",
         memory="trạng thái đã biết của hai ô và số bước kể từ lần quan sát gần nhất của từng ô",
         rules=["suy giảm niềm tin: p(bẩn) = 1 − (1 − r)^k, với r = 0,10 và k là số bước chưa quan sát",
                "lợi ích kỳ vọng = p(bẩn) × số bước còn lại trong tầm nhìn",
                "lợi ích kỳ vọng > chi phí di chuyển → di chuyển; ngược lại → NoOp"],
         note="tiêu chí quyết định là giá trị kỳ vọng của hàm lợi ích, không phải điều kiện đạt mục đích",
         limit="Tham số r cố định và giống nhau cho mọi ô. Trong cấu hình này ô B bẩn lại nhanh "
               "gấp năm lần ô A, nhưng tác tử không ước lượng lại r từ dữ liệu quan sát."),
    dict(en="learning agent",
         sense="cảm nhận về môi trường và phần thưởng thu được sau mỗi bước",
         memory="mô hình môi trường: ước lượng tốc độ bẩn lại r của từng ô",
         rules=["bộ phận phê bình: đọc phần thưởng so với chuẩn hiệu năng",
                "bộ phận học: cập nhật r(A) và r(B) từ dữ liệu quan sát",
                "bộ phận thực thi: so sánh lợi ích kỳ vọng với chi phí, dùng r đã ước lượng",
                "bộ sinh vấn đề: sinh hành động thăm dò với tỉ lệ giảm dần theo số bước"],
         note="bốn thành phần theo sơ đồ tác tử có khả năng học ở slide 19",
         limit="Ước lượng chỉ hội tụ sau một số bước đủ lớn. Trong giai đoạn đầu, hiệu năng "
               "thấp hơn tác tử hướng lợi ích."),
    dict(en="reinforcement learning · Q-learning",
         sense="cảm nhận về môi trường và phần thưởng thu được sau mỗi bước",
         memory="bảng Q kích thước 4 trạng thái × 3 hành động",
         rules=["Q(s, a) ← Q(s, a) + α · [ r + γ · max Q(s′, a′) − Q(s, a) ],&nbsp; α = 0,30,&nbsp; γ = 0,90",
                "chọn hành động cực đại Q tại trạng thái hiện tại",
                "thăm dò ε-greedy: với xác suất ε chọn hành động ngẫu nhiên, ε giảm dần"],
         note="học giá trị hành động trực tiếp từ phần thưởng, không xây dựng mô hình môi trường",
         limit="Bảng Q không biểu diễn mô hình môi trường. Không gian trạng thái chỉ gồm bốn "
               "cảm nhận, tương ứng môi trường quan sát bộ phận; chính sách học được do đó không tối ưu."),
]

CSS = """
<style>
.block-container{padding-top:2.2rem;max-width:1280px}
.vw-world{display:grid;grid-template-columns:1fr 1fr;border:2px solid #00506F;max-width:620px}
.vw-cell{position:relative;height:180px;padding:7px 9px;border-right:2px solid #00506F}
.vw-cell:last-child{border-right:0}
.vw-here{background:#F7FBFD}
.vw-nm{font:italic 20px/1 'Times New Roman',serif;color:#00506F}
.vw-dirt{position:absolute;left:14%;bottom:10px;width:38%}
.vw-mach{position:absolute;right:7%;top:34px;width:46%}
.vw-suck{position:absolute;left:50%;top:12px;transform:translateX(-50%);
  font-size:11px;font-weight:bold;color:#EC703D}
.vw-dt{font-size:11px;color:#006A98;text-transform:uppercase;letter-spacing:.04em;
  font-weight:bold;margin-top:11px}
.vw-dt:first-child{margin-top:0}
.vw-dd{margin:2px 0 0;font-size:14px;color:#3D3D3D}
.vw-dd ul{margin:2px 0 0;padding-left:19px}
.vw-note{color:#7A7A7A;font-size:11.5px;margin-top:4px}
.vw-limit{background:#FFF6F1;border-left:3px solid #EC703D;padding:8px 11px;
  margin-top:13px;font-size:13px;color:#3D3D3D}
.vw-limit b{color:#B44B1E}
.vw-live{background:#F2F7FA;border-left:3px solid #4590B8;padding:8px 11px;margin-top:4px;
  font-family:Consolas,Menlo,monospace;font-size:12px;white-space:pre-wrap;color:#3D3D3D}
</style>
"""

HERE = Path(__file__).parent
MACHINE = (HERE / "machine.svg").read_text(encoding="utf-8")
DIRT = (HERE / "dirt.svg").read_text(encoding="utf-8")


def percept_vi(p):
    return "[%s, %s]" % (p[0], VI_STATUS[p[1]])


def world_html(world, last_action):
    cells = []
    for c in ag.CELLS:
        here = world.pos == c
        klass = "vw-cell" + (" vw-here" if here else "")
        parts = ['<span class="vw-nm">%s</span>' % c]
        if here and last_action == ag.SUCK:
            parts.append('<span class="vw-suck">đang hút</span>')
        if here:
            parts.append('<div class="vw-mach">%s</div>' % MACHINE)
        if world.dirt[c]:
            parts.append('<div class="vw-dirt">%s</div>' % DIRT)
        cells.append('<div class="%s">%s</div>' % (klass, "".join(parts)))
    return '<div class="vw-world">%s</div>' % "".join(cells)


def spec_html(i):
    s = SPEC[i]
    return (
        '<div class="vw-dt">Cảm nhận <i>(percept)</i></div><div class="vw-dd">%s</div>'
        '<div class="vw-dt">Trạng thái bên trong <i>(internal state)</i></div><div class="vw-dd">%s</div>'
        '<div class="vw-dt">Chương trình tác tử <i>(agent program)</i></div>'
        '<div class="vw-dd"><ul>%s</ul><div class="vw-note">%s</div></div>'
        '<div class="vw-limit"><b>Hạn chế.</b> %s</div>'
    ) % (s["sense"], s["memory"], "".join("<li>%s</li>" % r for r in s["rules"]),
         s["note"], s["limit"])


def belief_line(b):
    return ",   ".join("%s = %s" % (c, VI_STATUS[b[c]]) for c in ag.CELLS)


def explain(i, agent):
    """Diễn giải bước vừa thực hiện, dựng từ số liệu agent.detail."""
    d = getattr(agent, "detail", None)
    if not d:
        return "Chưa thực hiện bước nào."
    act = VI_ACTION.get(d.get("action"), "—")
    if i == 0:
        if d["matched"]:
            seq = "".join("[%s, %s]" % (c, VI_STATUS[s]) for c, s in d["matched"])
            return ("chuỗi cảm nhận đã nhận: %d mục\nhậu tố khớp trong bảng: %s\nhành động: %s"
                    % (d["seen"], seq, act))
        return "không có mục nào khớp\nhành động: %s" % act
    if i == 1:
        extra = "" if d["status"] == ag.DIRTY else "\ntrạng thái ô còn lại: không quan sát được, không lưu"
        return "luật khớp: [%s] → %s%s" % (VI_STATUS[d["status"]], act, extra)
    if i == 2:
        why = {"dirty": "ô hiện tại bẩn",
               "both known clean": "cả hai ô đã biết là sạch",
               "other cell unknown": "trạng thái ô còn lại chưa xác định"}[d["why"]]
        return ("trạng thái bên trong: %s\nđiều kiện khớp: %s\nhành động: %s"
                % (belief_line(d["belief"]), why, act))
    if i == 3:
        head = "trạng thái bên trong: %s" % belief_line(d["belief"])
        if d["reached"]:
            return head + "\nmục đích đã đạt, kế hoạch rỗng\nhành động: %s" % act
        plan = " → ".join(VI_ACTION[a] for a in d["plan"]) or "không sinh được kế hoạch"
        return (head + "\ntìm kiếm mở rộng %d trạng thái\nkế hoạch: %s"
                "\nhành động (bước đầu của kế hoạch): %s" % (d["expanded"], plan, act))
    if i == 4:
        if d["dirty_here"]:
            return ("ô hiện tại bẩn, hành động Hút bụi không phát sinh chi phí di chuyển"
                    "\nhành động: %s" % act)
        return ("p(bẩn | %s) = 1 − (1 − 0,10)^%d = %.2f"
                "\nlợi ích kỳ vọng = %.2f × %d = %.1f"
                "\nchi phí di chuyển = %d"
                "\nso sánh: lợi ích %s chi phí\nhành động: %s"
                % (d["other"], d["k"], d["p"], d["p"], d["span"], d["gain"], d["cost"],
                   ">" if d["gain"] > d["cost"] else "≤", act))
    if i == 5:
        head = ("mô hình đã học: r(A) ≈ %.3f   r(B) ≈ %.3f"
                "\ngiá trị thật của môi trường: r(A) = 0,040   r(B) = 0,200"
                "\nphần thưởng bước trước: %+d" % (d["rate_a"], d["rate_b"], d["critic"]))
        if d["mode"] == "suck":
            return head + "\nô hiện tại bẩn\nhành động: %s" % act
        if d["mode"] == "explore":
            return head + "\nbộ sinh vấn đề sinh hành động thăm dò, ε = %.2f\nhành động: %s" % (
                d["eps"], act)
        return head + ("\np(bẩn | %s) = %.2f  tính theo r đã ước lượng"
                       "\nlợi ích kỳ vọng = %.1f   chi phí di chuyển = %d"
                       "\nhành động: %s" % (d["other"], d["p"], d["gain"], d["cost"], act))
    lines = []
    if d["update"]:
        s0, a0, old, new, r = d["update"]
        lines.append("cập nhật Q[%s, %s] · %s: %.2f → %.2f   phần thưởng r = %+d"
                     % (s0[0], VI_STATUS[s0[1]], Q_ACTION[a0], old, new, r))
    st_, row = d["state"], d["row"]
    lines.append("Q[%s, %s] = %s" % (st_[0], VI_STATUS[st_[1]],
                 "  |  ".join("%s %.2f" % (Q_ACTION[k], v) for k, v in enumerate(row))))
    lines.append("ε = %.2f  → %s" % (d["eps"],
                 "chọn hành động thăm dò" if d["explore"] else "chọn hành động cực đại Q"))
    lines.append("chính sách hiện tại: " + ",  ".join(
        "[%s, %s] → %s" % (s[0], VI_STATUS[s[1]], Q_ACTION[agent.greedy(s)])
        for s in sorted(agent.Q)))
    return "\n".join(lines)


# ------------------------------------------------------------------ trạng thái
st.set_page_config(page_title="Mô phỏng tác tử — Chương 2", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

if "saved" not in st.session_state:
    st.session_state.saved = []
    st.session_state.running = False


def reset():
    c = st.session_state
    c.world = ag.World(dynamic=c.get("dyn", False), seed=c.get("seed", 0),
                       move_cost=c.get("cost", 1))
    c.agent = ag.AGENTS[c.pick](move_cost=c.get("cost", 1), seed=c.get("seed", 0) + 100)
    c.step = 0
    c.last = None
    c.trace = []
    c.running = False


def step_once():
    c = st.session_state
    if c.step >= 400:
        c.running = False
        return
    p = c.world.percept()
    a = c.agent(p)
    c.world.step(a)
    if hasattr(c.agent, "reward"):
        c.agent.reward(c.world.score)
    c.step += 1
    c.last = a
    c.trace.append({"Bước": c.step, "Cảm nhận": percept_vi(p),
                    "Hành động": VI_ACTION[a], "Điểm": c.world.score})


st.title("Mô phỏng tác tử — thế giới hút bụi")
st.caption("Môi trường gồm hai ô A và B. Tác tử chỉ quan sát được ô hiện tại — môi trường "
           "quan sát bộ phận. Hàm đo hiệu năng: mỗi bước cộng 1 điểm cho mỗi ô đang sạch, "
           "mỗi hành động di chuyển trừ đúng chi phí di chuyển.")

pick = st.segmented_control("Loại tác tử", options=list(range(7)),
                            format_func=lambda i: VI_NAME[i], default=0, key="pick")
if pick is None:
    st.session_state.pick = pick = 0

if "world" not in st.session_state or st.session_state.get("built") != pick:
    reset()
    st.session_state.built = pick

c1, c2, c3, c4 = st.columns([2.4, 2.2, 1.3, 1.6])
c1.checkbox("Bụi rơi lại (môi trường động)", key="dyn", on_change=reset)
c2.slider("Chi phí di chuyển", 0, 5, 1, key="cost", on_change=reset)
c3.number_input("Seed", 0, 999, 0, key="seed", on_change=reset)
c4.slider("Tốc độ (bước/giây)", 1, 10, 4, key="spd")

b1, b2, b3, _ = st.columns([1.2, 1.2, 1.2, 4.4])
if b1.button("Một bước", type="primary", use_container_width=True):
    step_once()
label = "Dừng" if st.session_state.running else "Tự chạy"
if b2.button(label, use_container_width=True):
    st.session_state.running = not st.session_state.running
if b3.button("Đặt lại", use_container_width=True):
    reset()

S = st.session_state
st.markdown(world_html(S.world, S.last), unsafe_allow_html=True)
m1, m2, m3, _ = st.columns([1, 1, 1, 5])
m1.metric("Bước", S.step)
m2.metric("Điểm hiệu năng", S.world.score)
m3.metric("Lần di chuyển", S.world.moves)

left, right = st.columns(2)
with left:
    st.subheader("Đặc tả tác tử — %s" % SPEC[pick]["en"])
    st.markdown(spec_html(pick), unsafe_allow_html=True)
    st.markdown('<div class="vw-dt">Suy diễn tại bước %s</div>'
                % (S.step if S.last else "—"), unsafe_allow_html=True)
    st.markdown('<div class="vw-live">%s</div>' % explain(pick, S.agent),
                unsafe_allow_html=True)

with right:
    st.subheader("Lịch sử")
    if S.trace:
        st.dataframe(pd.DataFrame(S.trace).set_index("Bước"),
                     use_container_width=True,
                     height=min(330, 40 + 35 * len(S.trace)))
    else:
        st.info("Chưa thực hiện bước nào.")
    s1, s2, _ = st.columns([1.4, 1.4, 2])
    if s1.button("Lưu lần chạy", use_container_width=True) and S.step:
        S.saved.append({"Tác tử": VI_NAME[pick],
                        "Môi trường": "động" if S.dyn else "tĩnh",
                        "Chi phí": S.cost, "Bước": S.step,
                        "Điểm": S.world.score, "Di chuyển": S.world.moves})
    if s2.button("Xoá bảng lưu", use_container_width=True):
        S.saved = []
    if S.saved:
        df = pd.DataFrame(S.saved)
        best = df["Điểm"].max()
        st.dataframe(df.style.apply(
            lambda r: ["background-color:#EAF4F9;font-weight:bold" if r["Điểm"] == best else ""
                       for _ in r], axis=1),
            use_container_width=True, hide_index=True)
    else:
        st.caption("Chưa có lần chạy nào được lưu. Lưu kết quả của từng tác tử để đối chiếu "
                   "trên cùng một cấu hình môi trường.")

if S.running:
    step_once()
    time.sleep(1.0 / S.spd)
    st.rerun()
