"""生成测试用中文 PDF 语料：《星海科技内部知识库手册》。

之所以用合成语料而不是随便找一份公开 PDF：
RAG 评测要能判定"答对了没有"，就必须有一份**事实可枚举、且大模型先验里绝对没有**的文档。
这份手册里的所有数字（住宿标准、SLA、审批权限……）都是虚构且自洽的，
所以"纯大模型回答"必然要靠编，RAG 才能答准 —— 对比效果才有说服力。

用法:
    python scripts/make_sample_pdf.py [--out data/sample_knowledge_base.pdf]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

FONT_CANDIDATES = [
    ("MSYH", "C:/Windows/Fonts/msyh.ttc", 0),
    ("MSYH", "C:/Windows/Fonts/msyhl.ttc", 0),
    ("SIMSUN", "C:/Windows/Fonts/simsun.ttc", 0),
    ("SIMHEI", "C:/Windows/Fonts/simhei.ttf", -1),
]

TITLE = "星海科技内部知识库手册"
SUBTITLE = "版本号 v3.2 · 生效日期 2026-03-01 · 人力资源部与信息技术部联合维护"

# (章节标题, [段落...])
SECTIONS: list[tuple[str, list[str]]] = [
    (
        "第 1 章 总则与适用范围",
        [
            "本手册全称为《星海科技内部知识库手册》，版本号 v3.2，生效日期 2026-03-01，"
            "由人力资源部与信息技术部联合维护，自生效之日起替代 v3.1 版本。",
            "本手册适用于星海科技全体正式员工（工号以 XH 开头）及试用期员工。"
            "外包人员与实习生仅适用第 2 章与第 6 章，不适用第 4 章与第 5 章。",
            "本手册的解释权归人力资源部。如本手册内容与劳动合同不一致，以劳动合同为准；"
            "如与国家法律法规不一致，以国家法律法规为准。",
            "文档编号规则为 XH-KB-<章号>-<序号>，例如 XH-KB-02-003 表示第 2 章第 3 条。"
            "员工可通过内部 AI 网关「星海 Brain」检索本手册全文。",
        ],
    ),
    (
        "第 2 章 差旅与费用报销",
        [
            "国内出差住宿标准为：一线城市（北京、上海、广州、深圳）每晚 600 元，"
            "二线城市每晚 400 元，其他城市每晚 300 元。住宿费用凭发票在标准内据实报销。",
            "出差补贴按实际出差天数发放，标准为每天 120 元，出发当日与返回当日各按 0.5 天计算。"
            "补贴无需提供发票，随报销单一并发放。",
            "员工应在出差结束后 15 个工作日内通过「星海 OA - 费用中心」提交报销申请。"
            "超过 30 个工作日未提交的，视为自动放弃报销，公司不再受理。",
            "报销审批权限如下：单笔报销金额在 2000 元（含）以下的，由直属主管审批；"
            "超过 2000 元但不超过 20000 元的，需部门负责人审批；超过 20000 元的，需分管副总裁审批。",
            "报销发票必须为增值税专用发票或增值税普通发票，发票抬头应为「星海科技（上海）有限公司」，"
            "税号为 91310115MA1K7XXXXX。抬头或税号错误的发票一律退回重开。",
            "同城交通费按实际发生金额报销，但单次网约车费用超过 200 元的，需在报销单中备注事由。",
        ],
    ),
    (
        "第 3 章 请假与考勤",
        [
            "正式员工每年享有 10 天带薪年假。工作满 3 年后年假增加至 15 天，工作满 10 年后增加至 20 天。"
            "年假可跨年度结转，但最多结转 5 天，且须在次年 3 月 31 日前使用完毕。",
            "病假需提供二级甲等及以上医院出具的病假证明。全年累计病假在 5 个工作日以内不扣薪；"
            "超过 5 个工作日的部分，按基本工资的 60% 发放。",
            "事假为无薪假，单次申请不得超过 3 个工作日，全年累计不得超过 20 个工作日。"
            "超过 20 个工作日的事假需分管副总裁特批。",
            "所有请假须提前在 OA 系统提交并获得审批后方可离岗。紧急情况可先电话报备主管，"
            "但须在返回岗位后 1 个工作日内补交申请，逾期未补交的按旷工处理。",
            "法定节假日加班按国家规定支付加班费，或经员工本人同意后安排调休，调休须在 6 个月内使用完毕。",
        ],
    ),
    (
        "第 4 章 服务器与云资源申请",
        [
            "申请测试环境服务器需通过「星海云 - 资源申请」提交工单，标准配置为 4 核 CPU、16 GB 内存、200 GB SSD。"
            "如需更高配置，须在工单中说明业务理由并经部门负责人审批。",
            "测试环境服务器的默认租期为 30 天。到期前 3 天系统会向申请人发送续期提醒。"
            "连续 7 天未续期且无 CPU 使用率记录的实例将被自动回收，回收后数据不可恢复。",
            "生产环境资源申请必须附带变更单号，变更单号格式为 CHG-YYYYMMDD-XXX，"
            "并由 SRE 团队在 2 小时内响应。缺少变更单号的申请将被自动驳回。",
            "禁止在测试环境存储任何真实用户的手机号、身份证号、银行卡号等个人信息。"
            "测试所需数据必须来自数据脱敏平台，违反者按一级违规处理。",
            "云资源成本按月分摊到各部门，成本报表每月 5 日自动生成并推送至部门负责人邮箱。",
        ],
    ),
    (
        "第 5 章 信息安全与合规红线",
        [
            "一级违规包括但不限于：泄露用户隐私数据、私自将生产数据库导出至本地、绕过审批直接修改生产配置。"
            "一级违规首次发现即解除劳动合同，涉嫌违法的移交司法机关。",
            "二级违规包括：使用弱口令、在办公终端安装未经授权的软件、将内部文档转发至个人邮箱。"
            "二级违规首次发现给予书面警告，再次发现降级处理。",
            "所有员工必须每 90 天更换一次域账号密码。密码长度不少于 12 位，"
            "且须同时包含大写字母、小写字母、数字与特殊字符，且不得与前 3 次使用过的密码重复。",
            "办公终端必须开启全盘加密并安装终端管理客户端（EDR）。"
            "未安装 EDR 的终端禁止接入公司内网，网络准入系统会自动拦截。",
            "访问生产数据库必须通过堡垒机，操作全程录屏，录屏记录保存 180 天。",
        ],
    ),
    (
        "第 6 章 值班与故障响应（SLA）",
        [
            "P0 级故障指核心交易链路完全不可用。P0 级故障响应时间为 5 分钟，目标恢复时间为 1 小时，"
            "故障期间每 15 分钟向全员同步一次进展。",
            "P1 级故障指核心功能严重受损但存在降级方案。P1 级故障响应时间为 15 分钟，目标恢复时间为 4 小时。",
            "P2 级故障响应时间为 1 小时，目标恢复时间为 1 个工作日；"
            "P3 级故障响应时间为 1 个工作日，目标恢复时间由值班工程师与需求方协商确定。",
            "值班工程师交接班时间为每日 09:00 与 21:00，交接须在值班群内完成书面交接，"
            "未交接清楚的事项须电话确认。交接记录保存 90 天。",
            "所有 P0 与 P1 级故障必须在故障恢复后 3 个工作日内提交复盘报告（RCA），"
            "由 SRE 负责人审核后归档。",
        ],
    ),
    (
        "第 7 章 绩效考核",
        [
            "绩效考核周期为半年度，分别在每年 1 月与 7 月进行。考核结果分为 S、A、B、C、D 五档。",
            "绩效等级为 S 的员工占比不超过部门人数的 10%，年终奖系数为 3.0；"
            "A 档占比不超过 30%，系数为 2.0；B 档占比不超过 50%，系数为 1.0；"
            "C 档系数为 0.5；D 档无年终奖且进入为期 3 个月的绩效改进计划。",
            "连续两个考核周期被评为 D 档的员工，公司将依法解除劳动合同。"
            "绩效结果为 C 或 D 的员工，可在结果公示后 5 个工作日内向人力资源部提出申诉。",
        ],
    ),
    (
        "第 8 章 入职、离职与账号管理",
        [
            "新员工入职后 3 个工作日内，IT 会开通域账号、企业邮箱与 OA 账号；"
            "测试环境权限需由直属主管在入职后 5 个工作日内单独申请。",
            "员工离职时，所有账号在离职生效日当天 18:00 统一停用，"
            "工作数据由直属主管在 3 个工作日内完成交接与归档。",
            "内部 AI 网关「星海 Brain」支持 deepseek-chat 与 qwen-plus 两个模型。"
            "调用需申请 API Key，默认配额为每人每天 2000 次请求，超出配额需单独提工单申请。",
        ],
    ),
]


def register_font() -> str:
    """注册一个可用的中文字体，返回字体名。"""
    for name, path, index in FONT_CANDIDATES:
        p = Path(path)
        if not p.exists():
            continue
        try:
            if index >= 0:
                pdfmetrics.registerFont(TTFont(name, str(p), subfontIndex=index))
            else:
                pdfmetrics.registerFont(TTFont(name, str(p)))
            return name
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError(
        "未找到可用的中文字体（已尝试 微软雅黑/宋体/黑体）。"
        "请手动安装字体后修改 scripts/make_sample_pdf.py 中的 FONT_CANDIDATES。"
    )


def build_pdf(out_path: Path) -> Path:
    font = register_font()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        title=TITLE,
        author="星海科技",
    )

    h1 = ParagraphStyle("h1", fontName=font, fontSize=20, leading=28, alignment=1, spaceAfter=6)
    sub = ParagraphStyle(
        "sub", fontName=font, fontSize=9.5, leading=15, alignment=1, textColor=colors.HexColor("#666666")
    )
    h2 = ParagraphStyle(
        "h2", fontName=font, fontSize=13.5, leading=20, spaceBefore=14, spaceAfter=6,
        textColor=colors.HexColor("#1a3d6d"),
    )
    body = ParagraphStyle(
        "body", fontName=font, fontSize=10.5, leading=17.5, alignment=TA_LEFT, spaceAfter=5,
        firstLineIndent=21,
    )

    story = [Paragraph(TITLE, h1), Paragraph(SUBTITLE, sub), Spacer(1, 6 * mm)]
    for title, paras in SECTIONS:
        story.append(Paragraph(title, h2))
        for p in paras:
            story.append(Paragraph(p, body))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("（本手册为 RAG 演示用虚构文档，所有制度与数字均为虚构，不代表任何真实公司。）", sub))

    doc.build(story)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sample_knowledge_base.pdf", help="输出 PDF 路径")
    args = ap.parse_args()
    path = build_pdf(Path(args.out))
    total = sum(len(p) for _, p in SECTIONS)
    print(f"[ok] 已生成语料 PDF: {path}  ({len(SECTIONS)} 章 / {total} 条)")


if __name__ == "__main__":
    main()
