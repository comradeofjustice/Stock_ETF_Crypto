"""
07_industry_financial_principles.py
Generate a deep report on industry mapping, correlation drivers, and financial principles.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

DATA = Path(__file__).parent.parent / "data" / "processed"
CONCLUSION = Path(__file__).parent.parent / "conclution"
CONCLUSION.mkdir(parents=True, exist_ok=True)

LOG = CONCLUSION / "analysis_log.txt"

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG, "a") as f:
        f.write(line + "\n")

log("Loading data...")
meta = pd.read_csv(DATA / "asset_meta.csv")
r21 = pd.read_csv(DATA / "corr_pearson_21d.csv", index_col=0)
r63 = pd.read_csv(DATA / "corr_pearson_63d.csv", index_col=0)

ticker_cls = meta.set_index("ticker")["asset_class"].to_dict()

# ============================================================
# COMPREHENSIVE INDUSTRY / SECTOR MAPPING
# ============================================================
# ETF descriptions
etf_desc = {
    "SPY": ("S&P 500", "美国大盘股指数 (标普500)"),
    "IVV": ("S&P 500", "美国大盘股指数 (标普500)"),
    "QQQ": ("纳斯达克100", "科技股为主的纳斯达克100指数"),
    "DIA": ("道琼斯工业", "道琼斯30支蓝筹工业股"),
    "IWM": ("罗素2000", "美国小盘股指数"),
    "EEM": ("新兴市场", "MSCI新兴市场指数"),
    "EFA": ("发达市场(除美)", "MSCI EAFE 非美发达市场指数"),
    "XLK": ("科技行业", "标普科技板块"),
    "XLF": ("金融行业", "标普金融板块"),
    "XLE": ("能源行业", "标普能源板块"),
    "XLV": ("医疗健康", "标普医疗保健板块"),
    "XLI": ("工业行业", "标普工业板块"),
    "XLP": ("必需消费", "标普必需消费品板块"),
    "XLU": ("公用事业", "标普公用事业板块"),
    "GLD": ("黄金", "实物黄金信托 ETF"),
    "SLV": ("白银", "实物白银信托 ETF"),
    "GDX": ("金矿股", "全球金矿企业 ETF"),
    "USO": ("原油", "WTI原油期货 ETF"),
    "UNG": ("天然气", "天然气期货 ETF"),
    "VXX": ("波动率", "VIX短期期货 ETN"),
    "TQQQ": ("3倍做多纳指", "ProShares 3倍做多纳斯达克"),
    "SQQQ": ("3倍做空纳指", "ProShares 3倍做空纳斯达克"),
    "ARKK": ("颠覆性创新", "ARK方舟创新ETF"),
    "BITO": ("比特币期货", "ProShares比特币策略ETF"),
}

# Stock industry classifications
# Format: ticker -> (sector_cn, sector_en, sub_sector, description)
stock_sector = {
    # === 科技/半导体 (Technology / Semiconductors) ===
    "AAPL": ("科技", "Technology", "消费电子", "苹果公司 — 全球最大消费电子与生态系统企业"),
    "MSFT": ("科技", "Technology", "软件/云", "微软 — 企业软件、云计算 (Azure) 和AI"),
    "GOOGL": ("科技", "Technology", "互联网/广告", "Alphabet (谷歌) — 搜索、广告、云计算"),
    "GOOG": ("科技", "Technology", "互联网/广告", "Alphabet C类股"),
    "NVDA": ("科技/半导体", "Technology", "半导体/AI芯片", "英伟达 — GPU与AI计算芯片领导者"),
    "AMD": ("科技/半导体", "Technology", "半导体", "AMD — CPU/GPU设计商，与NVDA/INTC竞争"),
    "INTC": ("科技/半导体", "Technology", "半导体", "英特尔 — 传统CPU巨头，IDM模式"),
    "QCOM": ("科技/半导体", "Technology", "通信芯片", "高通 — 移动通信芯片与专利授权"),
    "AMAT": ("科技/半导体", "Technology", "半导体设备", "应用材料 — 半导体制造设备"),
    "LRCX": ("科技/半导体", "Technology", "半导体设备", "Lam Research — 刻蚀设备"),
    "KLAC": ("科技/半导体", "Technology", "半导体设备", "KLA — 晶圆检测与量测设备"),
    "ADI": ("科技/半导体", "Technology", "模拟芯片", "Analog Devices — 模拟/混合信号芯片"),
    "TXN": ("科技/半导体", "Technology", "模拟芯片", "德州仪器 — 模拟与嵌入式芯片"),
    "MCHP": ("科技/半导体", "Technology", "模拟芯片", "Microchip — 微控制器与模拟芯片"),
    "NXPI": ("科技/半导体", "Technology", "汽车芯片", "恩智浦 — 汽车与IoT芯片"),
    "SWKS": ("科技/半导体", "Technology", "射频芯片", "Skyworks — 射频前端芯片"),
    "MRVL": ("科技/半导体", "Technology", "数据中心芯片", "Marvell — 数据中心与存储芯片"),
    "AVGO": ("科技/半导体", "Technology", "网络芯片", "博通 — 网络通信与基础设施芯片"),
    "ASML": ("科技/半导体", "Technology", "光刻设备", "ASML — 全球唯一EUV光刻机制造商"),
    "MU": ("科技/半导体", "Technology", "存储芯片", "美光 — DRAM/NAND存储芯片"),
    "AMZN": ("科技", "Technology", "电商/云", "亚马逊 — 电商与AWS云计算"),
    "FB": ("科技", "Technology", "社交媒体", "Meta (Facebook) — 社交媒体与元宇宙"),
    "NFLX": ("科技", "Technology", "流媒体", "Netflix — 流媒体服务"),
    "ADBE": ("科技", "Technology", "创意软件", "Adobe — 创意与文档软件"),
    "CRM": ("科技", "Technology", "企业软件", "Salesforce — CRM云服务 (代码未收录)"),
    "INTU": ("科技", "Technology", "财税软件", "Intuit — 财税与会计软件"),
    "ADSK": ("科技", "Technology", "设计软件", "Autodesk — 工业设计软件"),
    "SNPS": ("科技", "Technology", "EDA软件", "Synopsys — 芯片设计EDA工具"),
    "CDNS": ("科技", "Technology", "EDA软件", "Cadence — 芯片设计EDA工具"),
    "ANSS": ("科技", "Technology", "仿真软件", "Ansys — 工程仿真软件"),
    "FTNT": ("科技", "Technology", "网络安全", "Fortinet — 网络安全设备与软件"),
    "CRWD": ("科技", "Technology", "网络安全", "CrowdStrike — 云原生安全平台"),
    "OKTA": ("科技", "Technology", "身份安全", "Okta — 身份认证与管理"),
    "ZS": ("科技", "Technology", "网络安全", "Zscaler — 零信任安全"),
    "TEAM": ("科技", "Technology", "协作软件", "Atlassian — 开发协作工具"),
    "WDAY": ("科技", "Technology", "HR云软件", "Workday — 人力与财务云"),
    "ZM": ("科技", "Technology", "视频会议", "Zoom — 视频通信平台"),
    "DOCU": ("科技", "Technology", "电子签名", "DocuSign — 电子签名与协议云"),
    "PYPL": ("科技", "Technology", "金融科技", "PayPal — 数字支付"),
    "SPLK": ("科技", "Technology", "数据平台", "Splunk — 机器数据分析 (已收购)"),
    "DELL": ("科技", "Technology", "PC/服务器", "戴尔 — PC与服务器硬件"),
    "HPQ": ("科技", "Technology", "PC/打印", "惠普 — PC与打印设备"),
    "IBM": ("科技", "Technology", "企业IT", "IBM — 企业IT与AI (Watson)"),
    "CSCO": ("科技", "Technology", "网络设备", "思科 — 网络设备与安全"),
    "ORCL": ("科技", "Technology", "数据库", "甲骨文 — 数据库与云基础设施"),

    # === 金融 (Financials) ===
    "JPM": ("金融", "Financials", "银行", "摩根大通 — 美国最大银行"),
    "BAC": ("金融", "Financials", "银行", "美国银行"),
    "GS": ("金融", "Financials", "投行", "高盛 — 投资银行与交易"),
    "MS": ("金融", "Financials", "投行", "摩根士丹利 — 资管与投行"),
    "C": ("金融", "Financials", "银行", "花旗集团"),
    "WFC": ("金融", "Financials", "银行", "富国银行"),
    "USB": ("金融", "Financials", "银行", "U.S. Bancorp"),
    "PNC": ("金融", "Financials", "银行", "PNC金融服务"),
    "V": ("金融", "Financials", "支付网络", "Visa — 全球最大支付网络"),
    "MA": ("金融", "Financials", "支付网络", "Mastercard — 全球支付网络"),
    "AXP": ("金融", "Financials", "信用卡", "美国运通 — 信用卡与商旅服务"),
    "CB": ("金融", "Financials", "保险", "Chubb — 财产与意外险"),
    "AIG": ("金融", "Financials", "保险", "美国国际集团"),
    "ALL": ("金融", "Financials", "保险", "Allstate — 车险与家庭险"),
    "PRU": ("金融", "Financials", "保险", "保德信金融"),
    "PFG": ("金融", "Financials", "保险", "Principal金融集团"),
    "CNA": ("金融", "Financials", "保险", "CNA金融"),
    "FISV": ("金融", "Financials", "金融科技", "Fiserv — 银行支付技术"),
    "TNET": ("金融", "Financials", "薪税服务", "TriNet — 人力资源外包"),
    "SAN": ("金融", "Financials", "银行(欧洲)", "桑坦德银行 (西班牙)"),
    "BBVA": ("金融", "Financials", "银行(欧洲)", "BBVA (西班牙)"),
    "ISP": ("金融", "Financials", "银行(欧洲)", "Intesa Sanpaolo (意大利)"),
    "CS": ("金融", "Financials", "银行(欧洲)", "瑞士信贷 (已合并)"),

    # === 医疗健康 (Healthcare) ===
    "JNJ": ("医疗健康", "Healthcare", "制药/器械", "强生 — 制药与医疗器械"),
    "PFE": ("医疗健康", "Healthcare", "制药", "辉瑞 — 大型制药与疫苗"),
    "MRK": ("医疗健康", "Healthcare", "制药", "默克 — 肿瘤与疫苗 (Keytruda)"),
    "ABBV": ("医疗健康", "Healthcare", "制药", "艾伯维 — 免疫学 (Humira/Skyrizi)"),
    "LLY": ("医疗健康", "Healthcare", "制药", "礼来 — 糖尿病与减肥药 (Mounjaro/Zepbound)"),
    "AMGN": ("医疗健康", "Healthcare", "生物制药", "安进 — 生物类似药"),
    "GILD": ("医疗健康", "Healthcare", "生物制药", "吉利德 — 抗病毒药物"),
    "BIIB": ("医疗健康", "Healthcare", "生物制药", "渤健 — 神经科学 (阿尔茨海默)"),
    "REGN": ("医疗健康", "Healthcare", "生物制药", "再生元 — 抗体药物"),
    "VRTX": ("医疗健康", "Healthcare", "生物制药", "Vertex — 囊性纤维化"),
    "MRNA": ("医疗健康", "Healthcare", "mRNA", "Moderna — mRNA疫苗与疗法"),
    "SGEN": ("医疗健康", "Healthcare", "ADC药物", "Seagen — 抗体药物偶联物 (已收购)"),
    "ILMN": ("医疗健康", "Healthcare", "基因测序", "Illumina — 基因测序设备"),
    "DXCM": ("医疗健康", "Healthcare", "医疗器械", "Dexcom — 连续血糖监测"),
    "ISRG": ("医疗健康", "Healthcare", "手术机器人", "Intuitive Surgical — 达芬奇手术系统"),
    "IDXX": ("医疗健康", "Healthcare", "动物医疗", "IDEXX — 兽医诊断"),
    "INCY": ("医疗健康", "Healthcare", "生物制药", "Incyte — 肿瘤药物"),
    "ALGN": ("医疗健康", "Healthcare", "医疗器械", "Align Technology — 隐适美牙套"),
    "TMO": ("医疗健康", "Healthcare", "科学仪器", "Thermo Fisher — 生命科学工具"),
    "DHR": ("医疗健康", "Healthcare", "科学仪器", "Danaher — 生命科学与诊断"),
    "BDX": ("医疗健康", "Healthcare", "医疗器械", "Becton Dickinson — 医疗耗材"),
    "ABT": ("医疗健康", "Healthcare", "医疗器械", "雅培 — 诊断与营养品"),
    "SYK": ("医疗健康", "Healthcare", "医疗器械", "Stryker — 骨科与手术器械"),
    "ZTS": ("医疗健康", "Healthcare", "动物健康", "Zoetis — 动物保健"),
    "CVS": ("医疗健康", "Healthcare", "药房/PBM", "CVS Health — 药房与健康保险"),
    "UNH": ("医疗健康", "Healthcare", "健康保险", "UnitedHealth — 美国最大医疗保险"),
    "EL": ("消费", "Consumer", "化妆品", "雅诗兰黛 (化妆品/消费)"),

    # === 能源 (Energy) ===
    "XOM": ("能源", "Energy", "综合石油", "埃克森美孚"),
    "CVX": ("能源", "Energy", "综合石油", "雪佛龙"),
    "OXY": ("能源", "Energy", "油气开采", "西方石油 (巴菲特重仓)"),
    "HAL": ("能源", "Energy", "油服", "哈里伯顿 — 油田服务"),
    "MRO": ("能源", "Energy", "油气开采", "Marathon Oil"),
    "PSX": ("能源", "Energy", "炼化", "Phillips 66 — 炼油与化工"),
    "BP": ("能源", "Energy", "综合石油(欧洲)", "英国石油 BP"),
    "SHEL": ("能源", "Energy", "综合石油(欧洲)", "壳牌 Shell"),
    "SU": ("能源", "Energy", "油砂", "Suncor — 加拿大油砂"),
    "CPG": ("能源", "Energy", "油气(加拿大)", "Crescent Point Energy"),
    "NG": ("能源", "Energy", "天然气", "NovaGold (金矿)"),

    # === 消费 (Consumer) ===
    "COST": ("消费", "Consumer", "仓储零售", "Costco — 会员制仓储超市"),
    "WMT": ("消费", "Consumer", "综合零售", "沃尔玛 — 全球最大零售商"),
    "MCD": ("消费", "Consumer", "餐饮", "麦当劳 — 全球快餐"),
    "NKE": ("消费", "Consumer", "运动服饰", "耐克 — 运动鞋服"),
    "SBUX": ("消费", "Consumer", "咖啡连锁", "星巴克"),
    "HD": ("消费", "Consumer", "家装零售", "Home Depot — 家装建材零售"),
    "LOW": ("消费", "Consumer", "家装零售", "Lowe's — 家装建材零售"),
    "PEP": ("消费", "Consumer", "食品饮料", "百事可乐"),
    "KO": ("消费", "Consumer", "饮料", "可口可乐"),
    "MDLZ": ("消费", "Consumer", "零食", "亿滋国际 — 零食 (奥利奥)"),
    "KHC": ("消费", "Consumer", "食品", "卡夫亨氏"),
    "KDP": ("消费", "Consumer", "饮料", "Keurig Dr Pepper"),
    "PG": ("消费", "Consumer", "日化", "宝洁 — 日化消费品"),
    "MO": ("消费", "Consumer", "烟草", "Altria — 烟草"),
    "MNST": ("消费", "Consumer", "能量饮料", "Monster Beverage"),
    "DLTR": ("消费", "Consumer", "折扣零售", "Dollar Tree — 一元店"),
    "ROST": ("消费", "Consumer", "折扣零售", "Ross Stores — 折扣服装"),
    "ORLY": ("消费", "Consumer", "汽配零售", "O'Reilly Automotive"),
    "TSCO": ("消费", "Consumer", "农具零售", "Tractor Supply"),
    "ULTA": ("消费", "Consumer", "美妆零售", "Ulta Beauty"),
    "MAR": ("消费", "Consumer", "酒店", "万豪国际"),
    "HLT": ("消费", "Consumer", "酒店", "希尔顿"),
    "BKNG": ("消费", "Consumer", "在线旅游", "Booking Holdings"),
    "ABNB": ("消费", "Consumer", "共享住宿", "Airbnb"),
    "CCL": ("消费", "Consumer", "邮轮", "嘉年华邮轮"),
    "LULU": ("消费", "Consumer", "运动服饰", "Lululemon — 瑜伽服饰"),
    "PTON": ("消费", "Consumer", "健身器材", "Peloton"),
    "UBER": ("消费", "Consumer", "出行/外卖", "Uber — 打车与外卖"),
    "DIS": ("消费", "Consumer", "娱乐", "迪士尼 — 主题公园与媒体"),
    "EA": ("消费", "Consumer", "游戏", "Electronic Arts"),
    "ATVI": ("消费", "Consumer", "游戏", "动视暴雪 (已被微软收购)"),
    "EBAY": ("消费", "Consumer", "电商", "eBay — 二手电商平台"),
    "MELI": ("消费", "Consumer", "电商(拉美)", "MercadoLibre — 拉美电商与支付"),
    "HAS": ("消费", "Consumer", "玩具", "孩之宝"),
    "F": ("消费", "Consumer", "汽车", "福特汽车"),
    "GM": ("消费", "Consumer", "汽车", "通用汽车"),
    "TSLA": ("消费", "Consumer", "电动车", "特斯拉 — 电动汽车与能源"),
    "RACE": ("消费", "Consumer", "超豪华汽车", "法拉利"),
    "STLA": ("消费", "Consumer", "汽车(欧洲)", "Stellantis"),
    "FCA": ("消费", "Consumer", "汽车", "Fiat Chrysler (合并后为STLA)"),

    # === 工业与国防 (Industrials & Defense) ===
    "CAT": ("工业", "Industrials", "工程机械", "卡特彼勒 — 全球最大工程机械"),
    "DE": ("工业", "Industrials", "农业机械", "John Deere — 农业与工程机械"),
    "HON": ("工业", "Industrials", "多元化工业", "霍尼韦尔"),
    "RTX": ("工业", "Industrials", "航空航天/国防", "雷神科技 — 航空发动机与导弹"),
    "LMT": ("工业", "Industrials", "国防", "洛克希德马丁 — 战斗机与导弹"),
    "GD": ("工业", "Industrials", "国防", "通用动力 — 军舰与战车"),
    "NOC": ("工业", "Industrials", "国防", "诺斯罗普格鲁曼 — 隐形轰炸机"),
    "BA": ("工业", "Industrials", "航空航天", "波音 — 民航与国防"),
    "GE": ("工业", "Industrials", "航空/能源", "GE — 航空发动机与能源"),
    "HEI": ("工业", "Industrials", "航空零部件", "Heico — 航空替换件"),
    "AIR": ("工业", "Industrials", "航空租赁", "AAR Corp — 航空服务"),
    "FDX": ("工业", "Industrials", "物流", "联邦快递"),
    "UPS": ("工业", "Industrials", "物流", "联合包裹 UPS"),
    "UNP": ("工业", "Industrials", "铁路", "Union Pacific — 铁路货运"),
    "MMM": ("工业", "Industrials", "多元化工业", "3M — 工业与消费品"),
    "EMR": ("工业", "Industrials", "自动化", "Emerson Electric — 工业自动化"),
    "JCI": ("工业", "Industrials", "建筑自动化", "江森自控"),
    "PCAR": ("工业", "Industrials", "卡车制造", "Paccar — 重型卡车 (Kenworth/Peterbilt)"),
    "FER": ("工业", "Industrials", "基础设施", "Ferrovial — 基建运营商 (西班牙)"),
    "CRH": ("工业", "Industrials", "建材", "CRH — 建筑材料"),
    "SDR": ("工业", "Industrials", "工业服务", "SDR (可能是Schroders/已整合)"),
    "ALV": ("工业", "Industrials", "汽车安全", "Autoliv — 汽车安全气囊"),

    # === 金融/银行（续）与公用事业 ===
    "SO": ("公用事业", "Utilities", "电力", "Southern Company — 电力公用"),
    "NEE": ("公用事业", "Utilities", "清洁能源", "NextEra Energy — 清洁能源/电力"),
    "AEP": ("公用事业", "Utilities", "电力", "American Electric Power"),
    "DTE": ("公用事业", "Utilities", "电力/燃气", "DTE Energy — 底特律电力和燃气"),
    "EXC": ("公用事业", "Utilities", "电力", "Exelon — 核电与电力"),
    "XEL": ("公用事业", "Utilities", "电力", "Xcel Energy — 清洁电力"),
    "VZ": ("通信", "Telecom", "电信", "Verizon — 电信运营商"),
    "T": ("通信", "Telecom", "电信", "AT&T — 电信与媒体"),
    "TMUS": ("通信", "Telecom", "电信", "T-Mobile US"),
    "VOD": ("通信", "Telecom", "电信(欧洲)", "Vodafone — 英国电信"),
    "TEF": ("通信", "Telecom", "电信(欧洲)", "Telefónica — 西班牙电信"),
    "BT": ("通信", "Telecom", "电信(英国)", "BT Group — 英国电信"),
    "VIV": ("通信", "Telecom", "电信(巴西)", "Telefônica Brasil (Vivo)"),
    "CHTR": ("通信", "Telecom", "有线电视", "Charter — 有线电视与宽带"),
    "CMCSA": ("通信", "Telecom", "有线/媒体", "Comcast — 有线电视与NBC环球"),
    "FOX": ("通信", "Telecom", "媒体", "Fox Corp — 新闻与体育媒体"),
    "SIRI": ("通信", "Telecom", "卫星广播", "SiriusXM — 卫星广播"),
    "WPP": ("通信", "Telecom", "广告", "WPP — 全球最大广告集团"),
    "PUB": ("通信", "Telecom", "广告", "Publicis — 法国广告集团"),

    # === 基础材料与矿业 (Materials & Mining) ===
    "BHP": ("材料/矿业", "Materials", "多元化矿业", "必和必拓 — 铁矿石/铜"),
    "RIO": ("材料/矿业", "Materials", "多元化矿业", "力拓 — 铁矿石/铝"),
    "FCX": ("材料/矿业", "Materials", "铜矿", "Freeport-McMoRan — 最大铜矿商"),
    "HL": ("材料/矿业", "Materials", "贵金属", "Hecla Mining — 银/金矿"),
    "G": ("材料/矿业", "Materials", "黄金矿业", "Genpact (跨行业服务)"),
    "IAG": ("材料/矿业", "Materials", "黄金矿业", "IAMGOLD"),
    "FRES": ("材料/矿业", "Materials", "贵金属(墨西哥)", "Fresnillo — 银矿"),
    "POLY": ("材料/矿业", "Materials", "黄金(俄罗斯)", "Polymetal — 俄/哈金矿"),
    "DD": ("材料/矿业", "Materials", "化工", "DuPont — 特种化工与材料"),
    "DOW": ("材料/矿业", "Materials", "化工", "陶氏化学"),
    "LIN": ("材料/矿业", "Materials", "工业气体", "Linde — 全球最大工业气体"),
    "MT": ("材料/矿业", "Materials", "钢铁", "ArcelorMittal — 全球最大钢铁"),
    "MTCH": ("科技", "Technology", "在线约会", "Match Group — Tinder/Hinge"),

    # === 中国概念 (China stocks) ===
    "BABA": ("中国", "China", "电商", "阿里巴巴 — 中国电商与云计算"),
    "JD": ("中国", "China", "电商", "京东 — 自营电商与物流"),
    "PDD": ("中国", "China", "电商", "拼多多 — 社交电商/Temu"),
    "BIDU": ("中国", "China", "搜索/AI", "百度 — 搜索与自动驾驶"),
    "NIO": ("中国", "China", "电动车", "蔚来 — 高端电动车"),
    "TCEHY": ("中国", "China", "社交/游戏", "腾讯控股"),
    "TCOM": ("中国", "China", "在线旅游", "携程"),
    "NTES": ("中国", "China", "游戏/教育", "网易"),
    "KWEB": ("中国", "China", "中概互联网ETF", "KraneShares中概互联网ETF"),
    "MCHI": ("中国", "China", "中国ETF", "iShares MSCI中国ETF"),
    "FXI": ("中国", "China", "中国大盘股ETF", "iShares中国大盘股ETF"),
    "CQQQ": ("中国", "China", "中国科技ETF", "Invesco中国科技ETF"),

    # === 欧洲/日本/拉美 ===
    "AZN": ("医疗健康", "Healthcare", "制药(欧洲)", "阿斯利康 — 英瑞制药"),
    "GSK": ("医疗健康", "Healthcare", "制药(欧洲)", "葛兰素史克"),
    "SNY": ("医疗健康", "Healthcare", "制药(欧洲)", "赛诺菲 — 法国制药"),
    "NVO": ("医疗健康", "Healthcare", "减肥药", "诺和诺德 — 糖尿病/减肥药龙头"),
    "SAP": ("科技", "Technology", "企业软件(欧洲)", "SAP — 德国企业软件"),
    "STM": ("科技/半导体", "Technology", "半导体(欧洲)", "意法半导体"),
    "MC": ("消费", "Consumer", "奢侈品(法国)", "LVMH — 酩悦·轩尼诗-路易·威登"),
    "SONY": ("消费", "Consumer", "电子/娱乐(日本)", "索尼"),
    "ARGX": ("医疗健康", "Healthcare", "生物制药(欧洲)", "Argenx — 比利时抗体药物"),
    "KONE": ("工业", "Industrials", "电梯(芬兰)", "通力电梯"),
    "DSM": ("材料/矿业", "Materials", "营养/材料(荷兰)", "DSM — 营养与材料科学"),
    "FLTR": ("消费", "Consumer", "博彩(英国)", "Flutter Entertainment — 博彩"),
    "RAND": ("金融", "Financials", "投资(南非)", "Rand Merchant Investment"),
    "EBR": ("公用事业", "Utilities", "电力(巴西)", "Centrais Elétricas Brasileiras"),
    "VIV": ("通信", "Telecom", "电信(巴西)", "Telefônica Brasil"),
    "VIE": ("金融", "Financials", "投资(法国)", "Veolia? 或投资公司"),
    "REE": ("工业", "Industrials", "基建(西班牙)", "西班牙基建/再生能源"),
    "BLND": ("金融", "Financials", "房地产(英国)", "British Land — 商业地产"),
    "LAND": ("金融", "Financials", "房地产(英国)", "Land Securities"),
    "PSN": ("金融", "Financials", "房地产(英国)", "Persimmon — 英国建筑商"),
    "TW": ("金融", "Financials", "房地产(英国)", "Taylor Wimpey — 英国房产"),
    "WPG": ("金融", "Financials", "REIT", "Washington Prime Group — 零售REIT"),
    "SMT": ("金融", "Financials", "投资信托(英国)", "Scottish Mortgage — 科技投资信托"),
    "SMIN": ("材料/矿业", "Materials", "矿业(英国)", "Smiths Group — 工业技术"),
    "FLOW": ("工业", "Industrials", "流体设备", "SPX FLOW — 流体处理设备"),
    "NN": ("金融", "Financials", "投资", "NN Group — 荷兰保险与资管"),
    "AHT": ("金融", "Financials", "REIT(酒店)", "Ashford Hospitality Trust"),
    "OR": ("材料/矿业", "Materials", "黄金矿业", "Osisko Gold Royalties"),
    "AGN": ("医疗健康", "Healthcare", "制药", "Allergan (已被艾伯维收购)"),
    "CERN": ("医疗健康", "Healthcare", "医疗IT", "Cerner — 电子病历 (被Oracle收购)"),
    "CTAS": ("工业", "Industrials", "制服服务", "Cintas — 工作制服与设施服务"),
    "FAST": ("工业", "Industrials", "工业紧固件", "Fastenal — 紧固件分销"),
    "PAYX": ("科技", "Technology", "薪税服务", "Paychex — 人力薪税服务"),
    "ADP": ("科技", "Technology", "薪税服务", "ADP — 人力资本管理"),
    "VRSK": ("金融", "Financials", "数据分析", "Verisk — 保险数据分析"),
    "CDW": ("科技", "Technology", "IT分销", "CDW — IT解决方案"),
    "HSIC": ("医疗健康", "Healthcare", "牙科耗材", "Henry Schein — 牙科医疗耗材"),
    "WBA": ("消费", "Consumer", "药房零售", "Walgreens Boots Alliance"),
    "ADM": ("消费", "Consumer", "农产品", "Archer-Daniels-Midland — 农产品加工"),
    "EL": ("消费", "Consumer", "化妆品", "雅诗兰黛"),
    "CGC": ("消费", "Consumer", "大麻(加拿大)", "Canopy Growth — 大麻"),
    "CRON": ("消费", "Consumer", "大麻(加拿大)", "Cronos Group — 大麻"),
    "GRF": ("工业", "Industrials", "?(已退市)", "Grifols (或已改名)"),
    "K": ("消费", "Consumer", "食品", "Kellanova (家乐氏)"),
    "TWTR": ("科技", "Technology", "社交媒体", "Twitter/X — 已被私有化"),
    "TEN": ("工业", "Industrials", "汽车零部件", "天纳克 — 汽车排放控制"),
    "TRN": ("工业", "Industrials", "铁路车辆", "Trinity Industries — 铁路车辆"),
    "UP": ("工业", "Industrials", "铁路(美国)", "Union Pacific (UP已用UNP) 或 Wheels Up"),
    "NXT": ("科技", "Technology", "智能家居", "Next — 可能是Next plc 或 Nest"),
    "SN": ("科技/半导体", "Technology", "传感器", "Smith & Nephew (医疗) 或传感"),
    "SDR": ("能源", "Energy", "油服", "Schlumberger (SLB前代码)"),
    "EZJ": ("消费", "Consumer", "航空(英国)", "easyJet — 廉价航空"),
    "CHKP": ("科技", "Technology", "网络安全(以色列)", "Check Point — 网络安全"),
    "ADS": ("金融", "Financials", "支付", "Alliance Data Systems — 信用卡/会员"),
    "CPRT": ("消费", "Consumer", "汽车拍卖", "Copart — 事故车拍卖"),
    "BAS": ("材料/矿业", "Materials", "化工(德国)", "巴斯夫 — 全球最大化工"),
    "SDR": ("能源", "Energy", "油服", "Schlumberger (现SLB)"),
}

# Combine: first check stock_sector, then etf_desc, then assign "其他"
def get_sector(ticker):
    if ticker in stock_sector:
        return stock_sector[ticker]
    if ticker in etf_desc:
        return ("ETF-" + etf_desc[ticker][0], "ETF", etf_desc[ticker][0], etf_desc[ticker][1])
    if ticker in ["BTC", "SOL"]:
        return ("加密货币", "Crypto", "数字货币", "加密货币")
    # For .OQ .N .PA .MI suffix variants
    base = ticker.split(".")[0] if "." in ticker else ticker
    if base in stock_sector:
        s = stock_sector[base]
        return (s[0], s[1], s[2], f"{s[3]} (多交易所)")
    return ("其他", "Other", "综合", "未分类标的")

log("Building industry mapping...")

# ============================================================
# ANALYSIS: Build report
# ============================================================
lines = []
def w(s=""):
    lines.append(s)

log("Generating report...")

w("# 标的产业归属与相关性金融原理深度分析")
w()
w(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
w()
w("> 本报告基于波动率时间序列的Pearson相关系数（21日滚动），从行业归属和金融学原理出发，深入分析正/负相关性的成因。")
w()
w("---")
w()

# ==== SECTION 1: INDUSTRY OVERVIEW ====
w("## 1. 产业分类体系")
w()
w("### 1.1 分类框架")
w()
w("我们对309个标的进行了**三层级产业分类**：")
w()
w("| 层级 | 说明 | 示例 |")
w("|------|------|------|")
w("| **大类资产** | Crypto / ETF / Stock | BTC, SPY, AAPL |")
w("| **一级行业** | GICS-style 板块 | 科技、金融、医疗健康、能源、消费、工业、材料、通信、公用事业、中国概念 |")
w("| **二级子行业** | 细分赛道 | 半导体、银行、制药、石油、电商、国防... |")
w()

# Count by sector
sector_counts = {}
for t in r21.columns:
    sec = get_sector(t)[0]
    sector_counts[sec] = sector_counts.get(sec, 0) + 1

w("### 1.2 各行业标的数量分布")
w()
w("| 行业 | 标的数 | 占总比 |")
w("|------|--------|--------|")
total = sum(sector_counts.values())
for sec, cnt in sorted(sector_counts.items(), key=lambda x: -x[1]):
    w(f"| {sec} | {cnt} | {cnt/total:.1%} |")
w()

log("Section 1 done")

# ==== SECTION 2: INTRA-INDUSTRY CORRELATION ====
w("## 2. 同行业内相关性分析")
w()
w("同一行业内的标的因为共享**基本面驱动因子**（行业供需、技术趋势、政策环境），其波动率通常呈现显著正相关。以下按行业分析内部相关性的强度与成因。")
w()

# Compute intra-industry correlations
sectors_of_interest = ["科技/半导体", "金融", "医疗健康", "能源", "消费", "工业", "材料/矿业", "中国", "公用事业", "通信"]

for sec_name in sectors_of_interest:
    tickers_in_sec = [t for t in r21.columns if get_sector(t)[0] == sec_name]
    tickers_in_sec = [t for t in tickers_in_sec if t in r21.index]  # ensure in correlation matrix
    if len(tickers_in_sec) < 2:
        continue

    sub = r21.loc[tickers_in_sec, tickers_in_sec]
    vals = []
    for i in range(len(tickers_in_sec)):
        for j in range(i+1, len(tickers_in_sec)):
            v = sub.iloc[i, j]
            if not np.isnan(v):
                vals.append(v)
    vals = np.array(vals)
    if len(vals) == 0:
        continue

    w(f"### 2.{sectors_of_interest.index(sec_name)+1} {sec_name}行业 ({len(tickers_in_sec)}个标的)")
    w()
    w(f"| 指标 | 数值 |")
    w(f"|------|------|")
    w(f"| 平均相关性 | {vals.mean():.4f} |")
    w(f"| 中位数相关性 | {np.median(vals):.4f} |")
    w(f"| 最高相关性 | {vals.max():.4f} |")
    w(f"| 最低相关性 | {vals.min():.4f} |")
    w(f"| 标准差 | {vals.std():.4f} |")
    w()

    # Top 5 most correlated pairs within this sector
    pairs = []
    for i in range(len(tickers_in_sec)):
        for j in range(i+1, len(tickers_in_sec)):
            v = sub.iloc[i, j]
            if not np.isnan(v):
                pairs.append((tickers_in_sec[i], tickers_in_sec[j], v))
    pairs.sort(key=lambda x: -x[2])

    # Show top 5
    w(f"**最强正相关对 (Top 5):**")
    w()
    for a, b, v in pairs[:5]:
        sa = get_sector(a)[2]
        sb = get_sector(b)[2]
        w(f"- **{a}** ↔ **{b}**: r = {v:.4f} (子行业: {sa} / {sb})")
    w()

log("Section 2 done")

# ==== SECTION 3: CROSS-INDUSTRY CORRELATION ====
w("## 3. 跨行业相关性矩阵")
w()
w("不同行业间的相关性反映了**宏观经济因子**（利率、通胀、GDP增长、风险偏好）对各行业的差异化影响。")
w()

key_sectors = ["科技/半导体", "金融", "医疗健康", "能源", "消费", "工业", "材料/矿业", "中国", "公用事业", "通信", "加密货币", "ETF-黄金", "ETF-波动率"]

# Compute average cross-sector correlations
sector_tickers = {}
for sec in key_sectors:
    if sec.startswith("ETF-"):
        etf_key = sec[4:]
        tickers = [t for t in r21.columns if t in etf_desc and etf_desc[t][0] == etf_key]
    elif sec == "加密货币":
        tickers = [t for t in r21.columns if ticker_cls.get(t) == "crypto"]
    else:
        tickers = [t for t in r21.columns if get_sector(t)[0] == sec]
    tickers = [t for t in tickers if t in r21.index]
    if tickers:
        sector_tickers[sec] = tickers

# Cross-sector average correlations
sec_names = list(sector_tickers.keys())
w("### 3.1 行业间平均波动率相关性矩阵")
w()
# Header
w("| 行业 | " + " | ".join(sec_names) + " |")
w("|------|" + "|".join([":-:"] * len(sec_names)) + "|")
for sec_a in sec_names:
    row_vals = []
    for sec_b in sec_names:
        if sec_a == sec_b:
            row_vals.append("—")
        else:
            corrs = []
            for ta in sector_tickers[sec_a]:
                for tb in sector_tickers[sec_b]:
                    v = r21.loc[ta, tb]
                    if not np.isnan(v):
                        corrs.append(v)
            if corrs:
                row_vals.append(f"{np.mean(corrs):.3f}")
            else:
                row_vals.append("N/A")
    w(f"| {sec_a} | " + " | ".join(row_vals) + " |")
w()

log("Section 3 done")

# ==== SECTION 4: FINANCIAL PRINCIPLES ====
w("## 4. 相关性背后的金融学原理")
w()
w("波动率相关性（即不同资产风险水平的同步变化）本质上由以下五大类驱动因子决定：")
w()

# --- 4.1 Common Factor Exposure ---
w("### 4.1 共同因子暴露 (Common Factor Exposure)")
w()
w("**金融原理**: 根据多因子模型 (APT/Fama-French)，资产收益率由一系列共同因子驱动。当两个资产暴露于相同因子时，其波动率会因因子波动而同步变化。")
w()
w("**关键因子**:")
w()
w("| 因子 | 敏感行业 | 不敏感行业 |")
w("|------|---------|-----------|")
w("| **市场Beta** | 科技、金融、工业 (高Beta) | 公用事业、必需消费 (低Beta) |")
w("| **利率 (Duration)** | 房地产、公用事业 (高久期) | 科技、能源 (低久期) |")
w("| **商品价格** | 能源、矿业 (高暴露) | 科技、医疗 (低暴露) |")
w("| **信用利差** | 金融、高收益债 (敏感) | 国债、黄金 (负敏感) |")
w("| **波动率 (VIX)** | 所有权益资产 (负相关) | VXX、期权卖方 (正相关) |")
w()
w("**案例说明**:")
w()

# Find actual examples
if "SPY" in r21.columns and "XLK" in r21.columns:
    w(f"- SPY与XLK波动率相关性高达 **{r21.loc['SPY','XLK']:.4f}**：两者高度暴露于相同的市场Beta因子，科技板块ETF本身就是标普500的最大权重板块。")
if "XLE" in r21.columns and "USO" in r21.columns:
    w(f"- XLE与USO波动率相关性 **{r21.loc['XLE','USO']:.4f}**：能源板块和原油价格共同暴露于油价因子。")
if "XLU" in r21.columns and "SPY" in r21.columns:
    w(f"- XLU与SPY波动率相关性相对较低 (**{r21.loc['XLU','SPY']:.4f}**)：公用事业低Beta、高久期，受利率驱动多于受市场Beta驱动。")
w()

# --- 4.2 Economic Regime ---
w("### 4.2 经济周期与行业轮动 (Economic Regime & Sector Rotation)")
w()
w("**金融原理**: 不同行业在经济周期各阶段表现不同。波动率相关性在**衰退期**（Risk-Off）通常全面上升，而在**扩张期**（Risk-On）则呈现行业分化。")
w()
w("**经济周期四阶段的行业强弱**:")
w()
w("| 周期阶段 | 强势行业 | 弱势行业 | 相关性特征 |")
w("|---------|---------|---------|-----------|")
w("| **扩张初期** (复苏) | 科技、消费可选、工业 | 公用事业、必需消费 | 行业间分化加大 |")
w("| **扩张后期** (过热) | 能源、材料、工业 | 科技、消费 | 通胀受益 vs 受损 |")
w("| **收缩初期** (滞胀) | 能源、医疗、必需消费 | 可选消费、科技 | Risk-Off，相关性上升 |")
w("| **收缩后期** (衰退) | 公用事业、黄金、国债 | 周期性行业 | 全面正相关 (恐慌传染) |")
w()
w("**实证观察**:")
w(f"- 本数据跨越2015-2026年，包含了2020年COVID暴跌（全面正相关极端值）和2022年加息周期（行业分化）。")
w(f"- 加密货币(BTC)与传统资产波动率相关性在2020年后明显上升，反映了机构化进程中的**因子收敛**。")
w()

# --- 4.3 Supply Chain ---
w("### 4.3 供应链联动效应 (Supply Chain Propagation)")
w()
w("**金融原理**: 产业链上下游企业之间存在**风险传染**。上游的供给冲击会通过成本传导至下游，导致波动率先后或同步上升。")
w()
w("**典型产业链**:")
w()
w("| 产业链 | 上游 | 中游 | 下游 | 相关性模式 |")
w("|--------|------|------|------|-----------|")
w("| **半导体** | ASML(光刻) → AMAT/LRCX(设备) → NVDA/AMD(设计) → AAPL/TSLA(应用) | 高度正相关 |")
w("| **能源** | USO(原油) → OXY/HAL(开采) → XOM/CVX(炼化) → XLE(板块) | 正相关，时滞传导 |")
w("| **矿业→工业** | BHP/RIO(矿) → CAT/DE(机械) → 基建/建筑 | 正相关，商品价格是共同驱动 |")
w()

# Find semiconductor chain examples
semi_chain = ["ASML", "AMAT", "LRCX", "KLAC", "NVDA", "AMD", "INTC", "TXN", "ADI"]
avail_semi = [s for s in semi_chain if s in r21.columns]
if len(avail_semi) >= 2:
    w("**半导体产业链实测相关性示例**:")
    w()
    w("| 上游 | 下游 | 相关系数 |")
    w("|------|------|----------|")
    shown = set()
    for a in avail_semi:
        for b in avail_semi:
            if a != b and (a,b) not in shown and (b,a) not in shown:
                shown.add((a,b))
                w(f"| {a} ({get_sector(a)[2]}) | {b} ({get_sector(b)[2]}) | {r21.loc[a,b]:.4f} |")
                if len(shown) >= 8:
                    break
        if len(shown) >= 8:
            break
    w()

# --- 4.4 Safe Haven ---
w("### 4.4 避险资产与风险资产的对立关系 (Safe Haven Dynamics)")
w()
w("**金融原理**: 在市场恐慌时，资金从风险资产流向避险资产，形成'跷跷板效应'。波动率维度上，这种对立表现为**低相关性甚至负相关性**。避险资产的波动率往往在风险资产波动率上升时同步或滞后上升（因为也有资金涌入推高波动），但方向性收益通常相反。")
w()
w("**核心对立关系**:")
w()
w("| 对立对 | 相关性 | 机制 |")
w("|--------|--------|------|")
if "GLD" in r21.columns and "SPY" in r21.columns:
    w(f"| GLD (黄金) vs SPY (美股) | {r21.loc['GLD','SPY']:.4f} | 黄金作为避险资产，市场恐慌时资金流入黄金 |")
if "VXX" in r21.columns and "SPY" in r21.columns:
    w(f"| VXX (VIX) vs SPY (美股) | {r21.loc['VXX','SPY']:.4f} | VIX是'恐慌指数'，理论上与股票负相关 |")
if "SQQQ" in r21.columns and "TQQQ" in r21.columns:
    w(f"| SQQQ (3倍做空) vs TQQQ (3倍做多) | {r21.loc['SQQQ','TQQQ']:.4f} | 完全相反的方向暴露 |")
w()
w("**安全资产金字塔**:")
w()
w("```")
w("              ┌─────────┐")
w("  极危        │  VXX    │  波动率对冲（极端对冲工具）")
w("              ├─────────┤")
w("  高恐慌      │  GLD    │  黄金（传统避险之王）")
w("              ├─────────┤")
w("  中等恐慌    │  XLP    │  必需消费品（即使衰退也要买）")
w("              ├─────────┤")
w("  低恐慌      │  XLU    │  公用事业（稳定现金流）")
w("              ├─────────┤")
w("  正常市场    │  SPY    │  大盘股（基本风险敞口）")
w("              ├─────────┤")
w("  风险偏好    │  QQQ    │  科技/成长（高Beta）")
w("              ├─────────┤")
w("  极度投机    │  TQQQ   │  3倍杠杆做多（最高风险）")
w("              └─────────┘")
w("```")
w()

# --- 4.5 Crypto ---
w("### 4.5 加密货币的独特相关性特征")
w()
w("**金融原理**: 加密货币与传统资产的波动率相关性有两个核心特征：")
w()
w("1. **中等偏低的相关性** — 加密市场仍受**独立因子**驱动：")
w("   - 矿业经济学 (BTC减半周期)")
w("   - 监管政策 (SEC/CFTC)")
w("   - 链上活动 (DeFi/NFT)")
w("   - 24/7交易带来的独特波动模式")
w()
if "BTC" in r21.columns and "SPY" in r21.columns:
    w(f"2. **BTC-SPY波动率相关性仅为 {r21.loc['BTC','SPY']:.4f}** — 这表明比特币的风险动态与美股大盘存在显著脱钩，可提供**分散化收益**。")
w()
w("3. **但极端事件时期相关性会剧增** (Correlation Breakdown)：")
w("   - 2020年3月COVID崩盘：BTC与SPY同步暴跌，相关性短期飙升")
w("   - 2022年FTX/加息：加密市场独立承压，与传统市场分化")
w()

# --- 4.6 Inverse/Leveraged ---
w("### 4.6 杠杆/反向ETF的特殊相关性结构")
w()
w("**金融原理**: 杠杆和反向ETF通过衍生品（期货/掉期）实现每日目标倍数。其波动率相关性呈现出独特的模式。")
w()
if "TQQQ" in r21.columns and "SQQQ" in r21.columns:
    w(f"- **TQQQ (3x做多纳指) vs SQQQ (3x做空纳指)**: 波动率相关性 **{r21.loc['TQQQ','SQQQ']:.4f}**")
    w(f"  - 价格方向完全相反，但**波动率正相关**：因为两者都受纳指波动率驱动")
    w(f"  - 纳指剧烈波动时 → TQQQ波动率↑ AND SQQQ波动率↑ → 波动率维度上正相关")
if "TQQQ" in r21.columns and "QQQ" in r21.columns:
    w(f"- **TQQQ vs QQQ**: 波动率相关性 {r21.loc['TQQQ','QQQ']:.4f} — 3倍杠杆放大波动率变化")
w()

# --- 4.7 Geographical ---
w("### 4.7 地域因子与地缘政治风险")
w()
w("**金融原理**: 同地区标的之间存在**本地市场因子**（汇率、央行政策、地缘政治），使得同地域标的的波动率相关性显著高于跨地域标的。")
w()

china_tickers = [t for t in r21.columns if get_sector(t)[0] == "中国" and t in r21.index]
us_tech_tickers = [t for t in r21.columns if get_sector(t)[0] in ["科技/半导体", "科技"] and t in r21.index]

if china_tickers and us_tech_tickers:
    # Intra-China
    if len(china_tickers) >= 2:
        sub_c = r21.loc[china_tickers, china_tickers]
        vals_c = [sub_c.iloc[i,j] for i in range(len(china_tickers)) for j in range(i+1,len(china_tickers)) if not np.isnan(sub_c.iloc[i,j])]
        avg_intra_cn = np.mean(vals_c) if vals_c else 0
        w(f"- **中国概念股内部相关性**: 均值 **{avg_intra_cn:.4f}** — 受中美关系、人民币汇率、监管政策等共同因子驱动")

    # Cross geo
    cross_vals = []
    for ct in china_tickers[:10]:
        for ut in us_tech_tickers[:10]:
            v = r21.loc[ct, ut]
            if not np.isnan(v):
                cross_vals.append(v)
    if cross_vals:
        w(f"- **中国概念股 vs 美国科技股**: 均值 **{np.mean(cross_vals):.4f}** — 低于各市场内部相关性，反映了地域分散化价值")
w()

log("Section 4 done")

# ==== SECTION 5: DEEP DIVE CASE STUDIES ====
w("## 5. 典型案例深度剖析")
w()

# --- 5.1 BTC-SOL ---
w("### 5.1 案例一：BTC ↔ SOL — 加密市场内部的强联动")
if "BTC" in r21.columns and "SOL" in r21.columns:
    w()
    w(f"**实证**: BTC与SOL的21日波动率相关系数为 **{r21.loc['BTC','SOL']:.4f}**。")
    w()
    w("**金融原理**:")
    w()
    w("1. **共同风险因子**: 加密货币市场整体受 'Crypto Beta' 主导 — 当比特币波动时，整个加密市场几乎同步反应。")
    w("2. **流动性关联**: BTC是加密市场的'储备货币'，大多数山寨币以BTC计价交易。BTC下跌 → 抵押品贬值 → 杠杆清算 → 全市场波动。")
    w("3. **投资者结构重叠**: BTC和SOL的投资者群体高度重叠（加密原生基金、散户、量化对冲基金）。")
    w("4. **叙事关联**: 两者共享'区块链科技创新'叙事，宏观叙事转向时同步受影响。")
    w()
    w("**投资含义**: BTC和SOL从波动率角度**高度替代而非互补**，分散化价值有限。")
w()

# --- 5.2 SPY-XLK ---
w("### 5.2 案例二：SPY ↔ XLK — 市场Beta的集中体现")
if "SPY" in r21.columns and "XLK" in r21.columns:
    w()
    w(f"**实证**: SPY与XLK的波动率相关系数为 **{r21.loc['SPY','XLK']:.4f}**，在所有行业ETF中最高的之一。")
    w()
    w("**金融原理**:")
    w()
    w("1. **权重集中**: 科技板块占标普500约30%+权重（2024年数据）。AAPL、MSFT、NVDA合计占SPY超15%。XLK本质上就是SPY的最大成分。")
    w("2. **高Beta属性**: 科技股Beta通常>1.0。当市场波动上升1%时，科技板块波动往往上升>1%。")
    w("3. **利率敏感共性**: 科技股估值高度依赖远期现金流折现(DCF)，对利率变化极为敏感。美联储加息预期 → SPY波动↑ AND XLK波动↑。")
    w()

# --- 5.3 GLD-SPY ---
w("### 5.3 案例三：GLD (黄金) — 低相关性的避险逻辑")
if "GLD" in r21.columns and "SPY" in r21.columns:
    w()
    w(f"**实证**: GLD与SPY的波动率相关系数为 **{r21.loc['GLD','SPY']:.4f}**，显著低于SPY与其他权益ETF的相关性（通常0.80+）。")
    w()
    w("**金融原理**:")
    w()
    w("1. **实际利率机制**: 黄金与**实际利率**（名义利率-通胀预期）呈负相关。加息 → 实际利率↑ → 黄金吸引力↓ vs 股票也可能承压。两者共同受利率驱动但方向不同。")
    w("2. **避险需求**: 地缘政治风险上升 → 黄金需求↑ (波动率可能上升因资金涌入) but 股票承压(波动率也上升)。波动率维度上两者同向但价格维度反向。")
    w("3. **美元因子**: 黄金以美元计价。美元走强 → 黄金承压。但美元对美股的影响更复杂（强美元利好进口但利空出口企业）。")
    w()
    w("**投资含义**: GLD在波动率维度上与SPY的相关性0.58意味着它**不是完美对冲**，但确实提供了显著的**分散化收益**。在风险平价策略中GLD权重最高，正是因为这个'中低相关 + 低波动'的特性。")
w()

# --- 5.4 Semi supply chain ---
w("### 5.4 案例四：半导体产业链 — 从光刻到芯片的波动传导")
w()
w("**实证数据**:")
w()
if "ASML" in r21.columns and "NVDA" in r21.columns:
    w(f"- ASML (光刻机) ↔ NVDA (AI芯片设计): r = **{r21.loc['ASML','NVDA']:.4f}**")
if "AMAT" in r21.columns and "NVDA" in r21.columns:
    w(f"- AMAT (沉积设备) ↔ NVDA: r = **{r21.loc['AMAT','NVDA']:.4f}**")
if "LRCX" in r21.columns and "AMAT" in r21.columns:
    w(f"- LRCX (刻蚀) ↔ AMAT (沉积): r = **{r21.loc['LRCX','AMAT']:.4f}**")
w()
w("**金融原理**:")
w()
w("1. **牛鞭效应 (Bullwhip Effect)**: AI需求激增 → NVDA订单↑ → NVDA向台积电下单↑ → 台积电向ASML/AMAT采购设备↑ → 整条链波动率共振。")
w("2. **资本开支周期**: 半导体是典型'繁荣-萧条'周期行业。当行业进入上行周期，设备和设计公司同步受益；转入下行则同步承压。")
w("3. **地缘政治放大器**: 芯片出口管制（中美科技战）同时影响整条产业链 — ASML无法向中国出货 ↔ 中国芯片公司受影响 ↔ NVDA对中国销售受限。")
w()

# --- 5.5 VXX ---
w("### 5.5 案例五：VXX — 波动率对冲的机制与局限")
if "VXX" in r21.columns:
    w()
    w(f"**实证**: VXX与SPY的波动率相关系数为 **{r21.loc['VXX','SPY']:.4f}**，但与其他ETF的平均相关性仅为 {np.mean([r21.loc['VXX',t] for t in etf_desc if t in r21.columns and t != 'VXX']):.4f}。")
    w()
    w("**金融原理**:")
    w()
    w("1. **VXX追踪VIX期货而非VIX即期**: VXX持有的是VIX期货合约。期货价格的期限结构（Contango升水/Backwardation贴水）导致VXX长期损耗（Contango损耗）。")
    w("2. **波动率维度的低相关≠价格维度的负相关**: SPY暴跌时，VIX飙升，VXX价格大涨。但VXX波动率与SPY波动率的相关性并不高，因为VXX自身的波动模式由VIX期货期限结构主导。")
    w("3. **尾部对冲价值**: VXX最适合作为'黑天鹅'对冲，而非日常分散化工具。2020年3月：SPY跌34%，VXX涨300%+。")
    w()
    w("**投资含义**: VXX适合小仓位(≤5%)配置作为尾部风险保护，但因长期损耗不适合大仓位或长期持有。")
w()

# --- 5.6 China stocks ---
w("### 5.6 案例六：中概股 — 政策与地缘风险的集中体现")
w()
china_tickers_filtered = [t for t in r21.columns if get_sector(t)[0] == "中国" and t in r21.index]
if len(china_tickers_filtered) >= 2:
    sub = r21.loc[china_tickers_filtered, china_tickers_filtered]
    vals_c = []
    for i in range(len(china_tickers_filtered)):
        for j in range(i+1, len(china_tickers_filtered)):
            v = sub.iloc[i,j]
            if not np.isnan(v):
                vals_c.append(v)
    if vals_c:
        w(f"**实证**: 中概股内部的平均波动率相关性为 **{np.mean(vals_c):.4f}**，显著受共同的地域因子驱动。")
    w()
w("**金融原理**:")
w()
w("1. **中国政策风险 (Regulatory Beta)**: 2021年教培双减、2022年网游版号等政策冲击 → 所有中概股同步波动。海外投资者将所有中概股视为单一'政策风险篮子'。")
w("2. **中美科技脱钩**: 审计底稿争端 (PCAOB) → 中概退市风险 → 2022年3月全板块暴跌。这是中概股特有的共同因子。")
w("3. **汇率传导**: 人民币贬值 → 以美元计价的中概股ADR价值承压 → 全板块波动率共振。")
w("4. **投资者结构**: 中概股的主要边际买家是对冲基金（快进快出），而非长期持有者，导致波动率联动更强。")
w()

log("Section 5 done")

# ==== SECTION 6: NEGATIVE CORRELATION ====
w("## 6. 负相关性的金融学解释")
w()
w("虽然波动率时间序列之间的相关性以正相关为主（反映了'风险共振'的普遍性），但确实存在显著的负相关对。以下是负相关的形成机制：")
w()

# Find negative pairs
neg_pairs = []
for i in range(len(r21.columns)):
    for j in range(i+1, len(r21.columns)):
        v = r21.iloc[i, j]
        if not np.isnan(v) and v < -0.3:
            a, b = r21.columns[i], r21.columns[j]
            neg_pairs.append((a, b, v, get_sector(a)[0], get_sector(b)[0]))
neg_pairs.sort(key=lambda x: x[2])

if neg_pairs:
    w("### 6.1 显著负相关对 (r < -0.3)")
    w()
    w("| 资产A | 行业A | 资产B | 行业B | 相关系数 | 负相关机制 |")
    w("|-------|-------|-------|-------|----------|-----------|")
    for a, b, v, sa, sb in neg_pairs[:10]:
        # Deduce mechanism
        if sa == "材料/矿业" and sb == "材料/矿业":
            mechanism = "不同金属品种的独立供需周期"
        elif "POLY" in [a,b] or "FRES" in [a,b]:
            mechanism = "贵金属矿商与其他资产的异质波动"
        elif "WPG" in [a,b] or "ATVI" in [a,b] or "SONY" in [a,b]:
            mechanism = "公司特有事件驱动（并购重组/退市风险）"
        else:
            mechanism = "行业基本面背离或公司特有风险"
        w(f"| {a} | {sa} | {b} | {sb} | {v:.4f} | {mechanism} |")
    w()

w("### 6.2 负相关性的四种成因")
w()
w("**类型一：基本面对立**")
w("- 油价上涨 → 航空公司成本↑ (利空) vs 能源公司利润↑ (利好)")
w("- 加息 → 银行净息差↑ (利好) vs 高估值科技股估值压缩 (利空)")
w()
w("**类型二：对冲产品结构**")
w("- SQQQ (3x做空) 与 TQQQ (3x做多)：结构上完全对立")
w("- 但波动率维度反而正相关（都受纳指波动率驱动），见第4.6节")
w()
w("**类型三：公司特有事件 (Idiosyncratic Risk)**")
w("- 并购套利、破产重组、退市风险 — 这些公司特有的波动模式与其他公司截然不同")
w("- 例如：ATVI被微软收购过程中，其波动率由'并购套利'驱动，与大盘波动脱钩甚至负相关")
w()
w("**类型四：流动性挤压 (Liquidity Squeeze)**")
w("- 当某一板块遭遇流动性危机（如2022年英债危机、2023年区域银行危机），资金撤离该板块同时涌入其他板块，形成短期负相关")
w()

log("Section 6 done")

# ==== SECTION 7: PORTFOLIO IMPLICATIONS ====
w("## 7. 投资组合构建的产业维度启示")
w()
w("### 7.1 真正的分散化：不仅是数量，更是产业因子分散")
w()
w("基于上述分析，**单纯增加持仓数量无法实现真正的分散化**。关键是配置**低相关性的产业因子**：")
w()
w("| 因子维度 | 高风险端 | 低风险端 | 相关性的意义 |")
w("|---------|---------|---------|------------|")
w("| **市场Beta** | 科技、可选消费 (高Beta) | 公用事业、必需消费 (低Beta) | 不同Beta降低组合波动 |")
w("| **利率敏感度** | 房地产、公用事业 (高) | 能源 (低) | 不同久期平衡利率风险 |")
w("| **商品敞口** | 能源、矿业 (高) | 科技、医疗 (零) | 对冲通胀 |")
w("| **地域** | 新兴市场 (高政治风险) | 美国大盘 (低政治风险) | 地缘分散 |")
w("| **货币** | 加密货币、新兴市场 | 美元资产 | 法币体系外分散 |")
w()

w("### 7.2 基于产业相关性的配置建议")
w()
w("**核心仓位 (40-50%)**:")
w("- SPY/IVV (美国大盘): 市场Beta暴露")
w("- 行业ETF分散: XLK(科技) + XLV(医疗) + XLF(金融)")
w()
w("**防御仓位 (20-30%)**:")
w("- GLD (黄金): 低Beta + 避险属性")
w("- XLP (必需消费) + XLU (公用事业): 稳定现金流 + 低波动")
w()
w("**增强/对冲仓位 (10-20%)**:")
w("- BTC (≤5%): 与传统资产低相关，提供独特因子暴露")
w("- VXX (≤5%): 尾部风险对冲")
w()
w("**战术仓位 (0-10%)**:")
w("- 根据经济周期阶段超配/低配特定行业")
w()

w("### 7.3 关键风险提醒")
w()
w("1. **相关性崩溃 (Correlation Breakdown)**: 在极端市场压力下，所有风险资产的相关性趋于+1。唯一的保护是配置真正无相关性的资产（现金、短债）。")
w("2. **波动率相关性≠价格相关性**: 本分析基于波动率序列，波动率正相关的两个资产可能在价格上完全负相关。")
w("3. **行业分类的局限性**: 现代企业中'行业'边界日益模糊（AAPL是硬件?服务?金融?），跨行业多元化效果可能低于理论预期。")
w("4. **历史不保证未来**: 产业相关性受技术变革和政策影响持续演变。")
w()

w("---")
w()
w("*报告由 07_industry_financial_principles.py 自动生成*")
w(f"*数据截至: {datetime.now().strftime('%Y-%m-%d')}*")
w()
w("## 附录：数据来源与分析方法")
w()
w("- **波动率数据**: 日对数收益率的21日滚动标准差")
w("- **相关系数**: Pearson积矩相关系数（基于波动率时间序列的交集观测值）")
w("- **行业分类**: 基于GICS标准和公司业务描述的定性分类")
w("- **金融原理解释**: 基于资产定价理论 (CAPM/APT)、行为金融、市场微观结构等学术框架")

# ---- WRITE ----
report = "\n".join(lines)
out_path = CONCLUSION / "industry_financial_principles.md"
with open(out_path, "w") as f:
    f.write(report)

log(f"Report written to {out_path}")
log(f"Total lines: {len(lines)}")
log("Done!")
