"""Build the strategy supplement and an updated, combined experiment PDF."""
from pathlib import Path
import argparse
import json
from xml.sax.saxutils import escape
import pandas as pd
from PIL import Image as PILImage
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether, PageBreak
from pypdf import PdfReader, PdfWriter

HERE=Path(__file__).resolve().parent
DEFAULT=HERE.parent if HERE.name=='code' else HERE/'outputs/BGCR_strategy_comparison_20261019/paper_report'
OLD=Path('E:/Users/24qxh/Desktop/TRMNL_BGCR_Experiments_20261019')


def main(root=DEFAULT):
    root=Path(root)
    for name,file in [('Song','simsun.ttc'),('Hei','simhei.ttf'),('TimesCustom','times.ttf'),('TimesCustomBold','timesbd.ttf')]:
        pdfmetrics.registerFont(TTFont(name,'C:/Windows/Fonts/'+file,**({'subfontIndex':0} if file.endswith('ttc') else {})))
    styles=getSampleStyleSheet()
    for name,font,size,leading,after in [('CN','Song',10,15,7),('SmallCN','Song',8.5,12,5),('TitleCN','Hei',23,30,12),
        ('H1CN','Hei',15,21,9),('H2CN','Hei',11,16,6),('TabHead','TimesCustomBold',8,10,0),
        ('TabCell','TimesCustom',8,10,0),('EnglishNote','TimesCustom',8,10.4,7)]:
        styles.add(ParagraphStyle(name,fontName=font,fontSize=size,leading=leading,spaceAfter=after,
            wordWrap='CJK' if font in ['Song','Hei'] else None,keepWithNext=name in ['H1CN','H2CN']))
    width=A4[0]-84
    tables={t['id']:t for t in json.loads((root/'tables/table_catalog.json').read_text(encoding='utf8'))}
    df=pd.read_csv(root/'data/comparison.csv');summary=pd.read_csv(root/'data/summary.csv')
    checks=json.loads((root/'data/checks.json').read_text(encoding='utf8'))
    n=len(df);variants=['R','A','B','C','D'];datasets=['OP','YELP','SO']
    def para(text,style='CN'):return Paragraph(escape(str(text)).replace('\n','<br/>'),styles[style])
    def text(story,s,style='CN'):story.append(KeepTogether([para(s,style)]))
    def title(story,s):story.append(para(s,'H1CN'))
    def rules_table(headers,rows,widths=None):
        data=[headers]+rows
        if widths is None:
            widths=[max(32,max(pdfmetrics.stringWidth(str(r[k]),'TimesCustom',8) for r in data)+14) for k in range(len(headers))]
            scale=min(1,width/sum(widths));widths=[w*scale for w in widths]
        data=[[para(v,'TabHead' if i==0 else 'TabCell') for v in r] for i,r in enumerate(data)]
        tab=Table(data,colWidths=widths,repeatRows=1,hAlign='CENTER')
        tab.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),4),
            ('BOTTOMPADDING',(0,0),(-1,-1),4),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),
            ('LINEABOVE',(0,0),(-1,0),.9,colors.black),('LINEBELOW',(0,0),(-1,0),.5,colors.black),
            ('LINEBELOW',(0,-1),(-1,-1),.9,colors.black)]))
        return tab
    def table(story,id):
        t=tables[id]
        story.append(KeepTogether([para(t['id']+'  '+t['title'],'H2CN'),rules_table(t['headers'],t['rows']),
            Spacer(1,5),para(t['note'],'EnglishNote'),Spacer(1,7)]))
    figures={21:('F21_strategy_total_runtime','构建加求解时间','点为各次独立运行；空心符号为中位数，误差棒为最小值至最大值。R为10个固定种子，A至D各为3次确定性进程重复。纵轴为对数。'),
        22:('F22_strategy_core_runtime','核心求解时间','与F21使用同一批结果，单独展示原生核心调用时间。策略所需的逆向CSR实际构建并计时，完整成本见F21。'),
        23:('F23_strategy_coverage_deficit','与精确覆盖边界的距离','纵轴为Kmax-K，越低越接近精确边界。请求未达到只说明本策略终止，不能作为原问题不可行的证书。'),
        24:('F24_strategy_score_gap','相同实际覆盖下的评分差距','每次运行均与自身实际K处的精确TPCAR评分比较。不同策略的终点不同，不能直接用其原始评分差判断优劣。')}
    def fig(story,num):
        name,caption,note=figures[num];path=root/'figures'/(name+'.png')
        with PILImage.open(path) as im:iw,ih=im.size
        w=min(width,173.8/25.4*72)
        story.append(KeepTogether([para('F%d  '%num+caption,'H2CN'),Image(str(path),width=w,height=w*ih/iw),
            para(note,'SmallCN'),Spacer(1,8)]))
    def footer(canvas,doc):
        canvas.saveState();canvas.setFont('Song',8);canvas.setFillColor(colors.HexColor('#555555'))
        canvas.drawString(42,25,'BGCR 同邻域策略比较补充实验 · 2026-10-19')
        canvas.setFont('TimesCustom',8);canvas.drawRightString(A4[0]-42,25,str(doc.page));canvas.restoreState()
    def build(path,story):
        SimpleDocTemplate(str(path),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=38,bottomMargin=40,
            title='BGCR selection-rule comparison',author='Experiment report').build(story,onFirstPage=footer,onLaterPages=footer)
    story=[]
    text(story,'BGCR同邻域策略比较','TitleCN')
    text(story,'补充随机扫描、原始A版及动态物品堆C版\n统一接口、独立重复与审稿意见对应说明')
    text(story,'本次在OP、Yelp、SO上重新运行R、A、B、C、D，共%d次独立进程运行；新增4张图和9份三线表。已有ID顺序扫描的9次历史结果单独保留，不计入本次样本。新版TPCAR继续作为同覆盖精确评分参照，未重跑旧版TPCAR或V0。'%n)
    title(story,'1 版本定义与可识别的比较')
    story.append(rules_table(['Version','Selection rule','Implementation'],[
        ['R','One random item permutation','Source aggregation'],
        ['A','Initial-loss order, one pass','Original donor-slot scan'],
        ['B','Same fixed order as A','Source aggregation'],
        ['C','Current global minimum loss','Source aggregation + item heap'],
        ['D','Same dynamic decisions as C','Source / terminal aggregation + user heap']],[38,214,width-252]))
    story.append(Spacer(1,9))
    text(story,'R/B用于评价按初始损失排序相对随机顺序的作用，两者共享供给聚合和物品查询。A/B保持交换规则不变，用于评价供给查询加速。B/C共享供给聚合与物品查询，比较固定单次扫描和动态最小损失选择。C/D保持动态交换规则不变，比较物品堆与用户堆及终端聚合的实现成本。')
    text(story,'因此，R/A不是完全隔离排序作用的主对照；C/D的速度差也不能全部归因于终端聚合。这里补齐的是策略对照，原先逐项关闭D版模块的消融继续保留。')
    title(story,'2 随机扫描的确切含义')
    text(story,'从同一Top-N初始解出发，对初始未覆盖且候选列非空的物品只随机排列一次。依次访问物品，在当前状态下选择引入该物品的最小损失可行直接替换；不存在则跳过。达到目标或扫描结束时停止。随机性只改变访问物品的先后顺序，用户及供给槽位仍按同一最小损失和同分规则选择。')
    text(story,'固定种子为2026101901至2026101910，并实际传入原生求解器。相同二进制与标准库下种子可重现；跨编译器的shuffle序列不作相同承诺。每次交换序列均保存，可核查实际执行过程。')

    story.append(PageBreak());title(story,'3 共同输入与计量口径')
    rows=[]
    for ds in datasets:
        g=df[df.dataset==ds].iloc[0];rows.append([ds,'10',str(int(g.initial_coverage)),str(int(g.D)),str(int(g.Kmax))])
    story.append(rules_table(['Dataset','N','K0','Request D','Exact Kmax'],rows))
    story.append(Spacer(1,10))
    text(story,'各版本使用相同graph_id、候选图、Top-N初始解、整数评分及取整规则。整数评分为q=rint(原始评分×1,000,000)，采用ties-to-even；每用户候选按q降序、物品ID升序排列。直接替换只能从当前覆盖次数至少为2的供给物品中移出一条推荐边，每次成功替换增加一个覆盖物品，用户推荐长度不变。')
    text(story,'最高请求统一为候选并集C，所有版本具有相同初始缺口D-K0。Yelp的C=7017而精确边界Kmax=7008，请求本身不可达；但启发式还会提前在更小覆盖处终止。这两种含义须分别说明。OP和SO的请求在全局可行，但受限于直接替换邻域的策略仍可能未达到。')
    text(story,'每个数据集R运行10个预先固定的种子，A/B/C/D各运行3次确定性进程重复。前3个区组内随机交错执行5个版本，之后完成剩余R种子。系统缓存未清空；独立进程不等于冷缓存。R的波动同时包含种子与机器计时波动，不能把其标准差解释为纯测量误差。')
    text(story,'构建加求解时间包含真实的方法专用输入构建和初始验证，加上核心调用；共享候选缓存准备不计入。末尾结果审计及导出另计，不混入核心时间。峰值为整个进程生命周期的OS工作集；记录了时间、内存采样、工作量计数、逐次交换和轨迹。预算沿用1800秒和12 GiB私有内存软守卫。')
    title(story,'4 已完成的一致性检查')
    text(story,'正式数据的每次返回方案均重新审计每用户长度、候选合法性、整数评分及覆盖。未达到目标的终点还检查是否已无可行直接替换。同一数据集的graph_id、初始覆盖、初始评分及D一致；A/B的交换事件逐项相同，C/D亦逐项相同，确定性重复的整数目标与覆盖不变。')
    text(story,'正式运行前另有120个小图、6061次原生调用验证，覆盖评分并列、负评分及多个N。动态选择与独立穷举当前最小损失步骤核对；随机扫描按给定物品次序的每一步与独立实现核对，并验证相同种子重现。另有6次完整子进程接口检查。这些校验不是正式66次样本，不重复计入统计。')
    if not all(c['same_inputs'] and len(c['equivalent_event_pairs'])==2 for c in checks):
        raise RuntimeError('Incomplete matched-input or event-equivalence checks')

    story.append(PageBreak());title(story,'5 时间比较')
    fig(story,21);fig(story,22)
    text(story,'同时查看两张图可区分构建成本与核心机制。只有目标达到情况和质量一起报告时，时间才有完整含义；动态最小损失并不保证最好终点或最快实现。','SmallCN')
    details=[]
    for ds in datasets:
        g=summary[summary.dataset==ds].set_index('variant')
        details.append('%s为C %.3f / D %.3f'%(ds,g.loc['C'].core_median,g.loc['D'].core_median))
    text(story,'C/D核心中位时间（秒）：'+'；'.join(details)+'。避免逆向索引构建带来的总时间收益，不等于原生核心在每个数据集上都更快。','SmallCN')
    story.append(PageBreak());title(story,'6 覆盖与质量比较')
    fig(story,23);fig(story,24)
    text(story,'误差棒均为观测最小值至最大值，不是置信区间。三线表同时给出均值与样本标准差。没有进行显著性推断，也没有将确定性重复当成独立问题实例。','SmallCN')

    for ds in datasets:
        story.append(PageBreak());title(story,'7.%d  %s结果三线表'%(datasets.index(ds)+1,'Yelp' if ds=='YELP' else ds))
        for stem in ['T13_quality_','T14_cost_','T15_work_']:table(story,stem+ds)
        g=summary[summary.dataset==ds].set_index('variant');r=g.loc['R'];b=g.loc['B'];d=g.loc['D']
        text(story,'本数据集R的平均终止覆盖为%.2f，样本标准差%.2f；B和D的覆盖分别为%.0f、%.0f。构建加求解中位时间分别为R %.3f秒、B %.3f秒、D %.3f秒。上述终点可能不同，质量差距已分别对齐精确TPCAR在各自实际覆盖处的评分。'%(r.coverage_mean,r.coverage_sd,b.coverage_mean,d.coverage_mean,r.total_median,b.total_median,d.total_median),'SmallCN')

    story.append(PageBreak());title(story,'8 如何写入正文和回应审稿意见')
    for ds in datasets:
        g=summary[summary.dataset==ds].set_index('variant');r=g.loc['R'];b=g.loc['B'];d=g.loc['D']
        text(story,'%s：R的平均覆盖%.2f，B为%.0f，D为%.0f；同覆盖评分差距均值分别为%.5f%%、%.5f%%、%.5f%%。'%(ds,r.coverage_mean,b.coverage_mean,d.coverage_mean,r.gap_mean,b.gap_mean,d.gap_mean),'SmallCN')
    text(story,'这组结果不支持预设“初始损失排序必然改善终点”或“动态贪心必然取得最好终点”。每步最小损失是一条局部选择规则；已有选择会影响后续供给，不能据此推出全程最优。D的实现价值应结合完整耗时和等价事件核对讨论，不应包装成质量全面优于随机扫描。','SmallCN')
    text(story,'正文建议把“选择策略对照”和“不改变规则的实现消融”分开叙述。先用R/B说明初始排序的经验作用，再用B/C讨论固定单次扫描相对动态最小损失选择的覆盖、质量和开销；A/B、C/D说明保持决策一致时实现结构的收益。原有1/2覆盖保证仍须依据邻域终止性质解释，不能从实验推导排序最优。')
    text(story,'可采用的英文实验说明','H2CN')
    text(story,'We compare five variants on OP, Yelp, and SO using the same candidate graph, initial Top-N solution, integer scores, and coverage target. R visits the initially uncovered items in one random permutation and applies the minimum-loss feasible replacement for each visited item. A uses the initial-loss order with a direct donor-slot scan, and B preserves the same decisions using source aggregation. C selects the current minimum-loss feasible replacement with an item heap, whereas D implements the same dynamic rule with a user heap and terminal aggregation. We use ten fixed seeds for R and three independent process runs for each deterministic variant. We report runtime together with attained coverage and the score gap to the exact solution at the same coverage.','EnglishNote')
    text(story,'可采用的英文回复草稿','H2CN')
    text(story,'We added a random-order scan and a dynamic minimum-loss baseline under a common experimental interface. The random scan changes only the item visitation order; the feasible replacement for each visited item is still chosen by its current minimum loss. Comparing R with B isolates the effect of the initial-loss order, while B versus C evaluates fixed single-pass selection against dynamic selection with the same source aggregation and item query. A/B and C/D additionally separate decision rules from implementation costs: their exchange sequences were verified to be identical within each pair. We now report attained coverage and score gaps at matched coverage alongside runtime, without attributing the coverage guarantee or uniform empirical superiority to the initial ordering.','EnglishNote')
    text(story,'上述草稿只回应本次补充的排序和单次扫描问题。隐式残量网络、精确性证明、TPCAR大缺口计算证据及BGCR理论边界，仍应按各自正文修订和既有实验逐项回应，不能声称这组比较已解决所有审稿意见。')
    title(story,'9 代码与复现入口')
    text(story,'PyCharm项目：E:/PythonProject。BGCR-STRATEGY-ALL运行本批全部设置；BGCR-STRATEGY-OP、YELP、SO为单数据集入口；BGCR-STRATEGY-RESUME只续跑缺失任务。BGCR-STRATEGY-PLOT只读取完成结果生成图表，不启动算法。')
    text(story,'代码目录：E:/PythonProject/BGCR_D。主入口run_strategy_comparison.py，执行与监控为strategy_worker.py及strategy_supervisor.py。每张图有独立figure_F21.py至figure_F24.py；strategy_report.py生成三线表和报告。数据目录保存66次正式结果的汇总、随机种子、版本定义、完整性检查及历史ID参照。')
    supplement=root/'BGCR_Strategy_Comparison_20261019.pdf';build(supplement,story)

    # Preserve the previously reviewed 37 pages, with an explicit updated cover.
    previous=OLD/'Experiment_Report_20261019.pdf'
    if previous.exists():
        cover=[];text(cover,'TPCAR与BGCR实验材料\n更新汇编','TitleCN')
        text(cover,'2026年10月19日 · 新增BGCR随机扫描与动态选择比较')
        text(cover,'本版保留上一版已经审阅的37页实验汇总，追加本次策略对照附录。旧报告中的“479次、20图、48表”仍是原部分的统计范围；新增部分为%d次、4图、9表。合并后共有%d条正式运行记录、24张数据图和57份三线表。'%(n,479+n))
        text(cover,'新增R/A/B/C/D对照在OP、Yelp、SO上使用共同候选图、初始解、整数评分和目标。R采用10个固定种子，A/B/C/D各3次进程重复；已有ID扫描的9条历史结果没有重复计数。旧版TPCAR及V0继续暂缓。')
        text(cover,'阅读说明：原部分的图F16至F18是早先Dynamic/Fixed/ID策略对照；新增F21至F24为本次重新运行的完整R/A/B/C/D对照，论文正式写作时优先使用新增结果。前后批计时不混合取最优，ID仅作单独标识的历史参照。')
        text(cover,'原部分及新增附录各保留自身页码。新增图均提供600 dpi彩色及灰度PNG、矢量PDF/SVG和逐图Python代码；新增三线表提供CSV、LaTeX及HTML。')
        cover_path=root/'qa/updated_cover.pdf';build(cover_path,cover)
        writer=PdfWriter()
        for name,path in [('更新说明',cover_path),('原实验汇总（479次）',previous),('新增策略比较（%d次）'%n,supplement)]:
            offset=len(writer.pages);reader=PdfReader(str(path));writer.append(reader,import_outline=False);writer.add_outline_item(name,offset)
        writer.add_metadata({'/Title':'TPCAR and BGCR experiments - updated strategy comparison','/Author':'Experiment report'})
        with (root/'Experiment_Report_20261019_v2.pdf').open('wb') as f:writer.write(f)
    print('PDF report generated',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=DEFAULT);args=parser.parse_args();main(args.root)
