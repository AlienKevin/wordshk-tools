use lazy_static::lazy_static;
use std::collections::HashSet;

lazy_static! {
    pub static ref ICONIC_SIMPS: HashSet<char> = {
        HashSet::from([
            '与', '专', '业', '丛', '东', '丝', '丢', '两', '严', '丧', '个', '临', '为', '丽',
            '举', '么', '义', '乌', '乐', '乔', '习', '乡', '书', '买', '乱', '争', '亏', '亚',
            '产', '亩', '亲', '亵', '亿', '仅', '从', '仑', '仓', '仪', '们', '众', '优', '会',
            '伛', '伞', '伟', '传', '伤', '伦', '伧', '伪', '伫', '体', '佥', '侠', '侣', '侥',
            '侦', '侧', '侨', '侩', '侪', '侬', '俣', '俦', '俨', '俩', '俪', '俫', '俭', '债',
            '倾', '偬', '偻', '偿', '傥', '傧', '储', '傩', '儿', '兖', '兰', '关', '兴', '兹',
            '养', '兽', '内', '冈', '册', '写', '军', '农', '冯', '冲', '决', '况', '冻', '净',
            '凄', '凉', '减', '凑', '凛', '凤', '凫', '凭', '凯', '击', '凿', '刍', '刘', '则',
            '刚', '创', '删', '别', '刭', '刹', '刽', '刿', '剀', '剂', '剐', '剑', '剥', '剧',
            '劝', '办', '务', '劢', '动', '励', '劲', '劳', '势', '勋', '匀', '匦', '匮', '区',
            '医', '华', '协', '单', '卖', '卢', '卤', '卫', '却', '卺', '厅', '历', '厉', '压',
            '厌', '厍', '厕', '厢', '厦', '厨', '厩', '厮', '县', '参', '双', '发', '变', '叙',
            '叠', '号', '叹', '叽', '吓', '吕', '吗', '吨', '听', '启', '吴', '呐', '呒', '呓',
            '呕', '呖', '呗', '员', '呛', '呜', '咏', '咙', '咛', '咝', '咤', '响', '哑', '哒',
            '哔', '哗', '哙', '哝', '哟', '唛', '唠', '唡', '唢', '唤', '啧', '啬', '啭', '啮',
            '啰', '啸', '喷', '喽', '喾', '嗳', '嘘', '嘤', '嘱', '噜', '嚣', '团', '园', '围',
            '囵', '国', '图', '圆', '圣', '圹', '场', '坏', '块', '坚', '坛', '坜', '坝', '坞',
            '坟', '坠', '垄', '垒', '垦', '垩', '垫', '垭', '垱', '垲', '埙', '埚', '堑', '堕',
            '墙', '壮', '声', '壳', '壶', '壸', '处', '备', '复', '够', '头', '夹', '夺', '奁',
            '奂', '奋', '奖', '奥', '妆', '妇', '妈', '妩', '妪', '妫', '姗', '娄', '娅', '娆',
            '娇', '娈', '娱', '娲', '娴', '婴', '婵', '婶', '嫒', '嫔', '嫱', '嬷', '孙', '学',
            '孪', '宁', '宝', '实', '宠', '审', '宪', '宫', '宽', '宾', '寝', '对', '寻', '导',
            '寿', '将', '尔', '尘', '尝', '尧', '尴', '尽', '层', '屃', '屉', '届', '属', '屡',
            '屿', '岁', '岂', '岖', '岗', '岘', '岚', '岛', '岭', '岿', '峄', '峡', '峣', '峥',
            '峦', '崂', '崃', '崭', '嵘', '嵚', '巅', '巩', '巯', '币', '帅', '师', '帏', '帐',
            '帜', '带', '帧', '帮', '帱', '帻', '帼', '幂', '并', '庄', '庆', '床', '庐', '庑',
            '库', '应', '庙', '庞', '废', '庼', '廪', '开', '异', '弃', '弑', '张', '弥', '弪',
            '弯', '弹', '强', '归', '当', '录', '彟', '彦', '彻', '径', '徕', '忆', '忏', '忧',
            '忾', '怀', '态', '怂', '怃', '怅', '怆', '怜', '总', '怼', '怿', '恋', '恒', '恳',
            '恶', '恸', '恺', '恻', '恼', '恽', '悫', '悬', '悭', '悯', '惊', '惧', '惨', '惩',
            '惫', '惬', '惭', '惮', '惯', '愤', '愦', '慑', '懑', '懒', '懔', '戆', '戋', '戏',
            '戗', '战', '戬', '扑', '执', '扩', '扪', '扫', '扬', '扰', '抚', '抛', '抟', '抠',
            '抡', '抢', '护', '报', '担', '拟', '拢', '拣', '拥', '拦', '拧', '拨', '择', '挚',
            '挛', '挝', '挞', '挟', '挠', '挡', '挢', '挣', '挤', '挥', '捞', '损', '捡', '换',
            '捣', '掳', '掴', '掷', '掸', '掺', '掼', '揽', '揿', '搀', '搁', '搂', '搅', '携',
            '摄', '摅', '摆', '摇', '摈', '摊', '撄', '撑', '撵', '撷', '撸', '撺', '擞', '攒',
            '敌', '敛', '数', '斋', '斓', '斩', '断', '无', '旧', '时', '旷', '旸', '昙', '昵',
            '昼', '显', '晋', '晒', '晓', '晔', '晕', '晖', '暂', '暧', '术', '机', '杀', '杂',
            '权', '条', '来', '杨', '构', '枞', '枢', '枣', '枥', '枧', '枨', '枪', '枫', '枭',
            '柠', '柽', '栀', '栅', '标', '栈', '栉', '栊', '栋', '栌', '栎', '栏', '树', '栖',
            '样', '栾', '桡', '桢', '档', '桤', '桥', '桦', '桧', '桨', '桩', '梦', '梼', '梾',
            '梿', '检', '棂', '椁', '椟', '椤', '椭', '椮', '楼', '榄', '榇', '榈', '榉', '槚',
            '槛', '槟', '横', '樯', '樱', '橥', '橱', '橹', '橼', '檩', '欢', '欤', '欧', '歼',
            '殁', '殇', '残', '殒', '殓', '殚', '殡', '殴', '毁', '毂', '毕', '毙', '毡', '毵',
            '气', '氢', '氩', '汇', '汉', '汤', '汹', '沟', '没', '沣', '沤', '沥', '沦', '沧',
            '沨', '沩', '沪', '泞', '泪', '泷', '泸', '泺', '泻', '泼', '泽', '泾', '洁', '洒',
            '洼', '浃', '浅', '浆', '浇', '浈', '浉', '浊', '测', '浍', '济', '浏', '浐', '浑',
            '浒', '浓', '浔', '涛', '涝', '涞', '涟', '涠', '涡', '涣', '涤', '润', '涧', '涨',
            '涩', '渊', '渌', '渍', '渎', '渐', '渑', '渔', '渖', '渗', '湾', '湿', '溃', '溅',
            '溆', '溇', '滚', '滞', '滟', '满', '滢', '滤', '滥', '滦', '滨', '滩', '潆', '潇',
            '潋', '潍', '潜', '潴', '澜', '濑', '濒', '灏', '灭', '灯', '灵', '灾', '灿', '炀',
            '炉', '炖', '炜', '炝', '点', '炼', '炽', '烁', '烂', '烃', '烛', '烟', '烦', '烧',
            '烨', '烩', '烫', '烬', '热', '焕', '焖', '焘', '爱', '爷', '牍', '牦', '牵', '牺',
            '犊', '状', '犷', '犸', '犹', '狈', '狝', '狞', '独', '狭', '狮', '狯', '狰', '狱',
            '狲', '猃', '猎', '猕', '猡', '猪', '猫', '猬', '献', '獭', '玑', '玙', '玚', '玛',
            '玮', '环', '现', '玱', '玺', '珐', '珑', '珰', '珲', '琎', '琏', '琐', '琼', '瑶',
            '瑷', '璎', '瓒', '瓮', '瓯', '电', '画', '畅', '畴', '疖', '疗', '疟', '疠', '疡',
            '疮', '疯', '疱', '痈', '痉', '痒', '痨', '痪', '痫', '瘅', '瘗', '瘘', '瘪', '瘫',
            '瘾', '瘿', '癞', '癣', '癫', '皑', '皱', '皲', '盏', '盐', '监', '盖', '盗', '盘',
            '眦', '睁', '睐', '睑', '瞒', '瞩', '矫', '矶', '矾', '矿', '砀', '码', '砖', '砗',
            '砚', '砜', '砺', '砻', '砾', '础', '硁', '硕', '硖', '硗', '硙', '硚', '碍', '碛',
            '碱', '礼', '祎', '祢', '祯', '祷', '祸', '禀', '禄', '禅', '离', '秃', '秆', '积',
            '称', '秽', '秾', '稣', '稳', '穑', '穷', '窃', '窍', '窎', '窑', '窜', '窝', '窥',
            '窦', '窭', '竖', '竞', '笃', '笋', '笔', '笕', '笺', '笼', '笾', '筚', '筛', '筜',
            '筝', '筹', '筼', '签', '简', '箓', '箦', '箧', '箨', '箩', '箪', '箫', '篑', '篓',
            '篮', '篯', '篱', '籁', '籴', '类', '籼', '粜', '粝', '粤', '粪', '粮', '紧', '絷',
            '纠', '纡', '红', '纣', '纤', '纥', '约', '级', '纨', '纩', '纪', '纫', '纬', '纭',
            '纮', '纯', '纰', '纱', '纲', '纳', '纴', '纵', '纶', '纷', '纸', '纹', '纺', '纻',
            '纽', '纾', '线', '绀', '绁', '绂', '练', '组', '绅', '细', '织', '终', '绉', '绊',
            '绌', '绍', '绎', '经', '绐', '绑', '绒', '结', '绔', '绕', '绖', '绗', '绘', '给',
            '绚', '绛', '络', '绝', '绞', '统', '绠', '绡', '绢', '绣', '绥', '绦', '继', '绩',
            '绪', '绫', '续', '绮', '绯', '绰', '绲', '绳', '维', '绵', '绶', '绷', '绸', '绹',
            '绺', '绻', '综', '绽', '绾', '绿', '缀', '缁', '缂', '缃', '缄', '缅', '缆', '缇',
            '缈', '缉', '缌', '缍', '缎', '缑', '缒', '缓', '缔', '缕', '编', '缗', '缘', '缙',
            '缚', '缛', '缜', '缝', '缟', '缠', '缡', '缢', '缣', '缤', '缥', '缦', '缧', '缨',
            '缩', '缪', '缫', '缬', '缭', '缮', '缯', '缰', '缱', '缳', '缴', '缵', '罂', '网',
            '罗', '罚', '罢', '罴', '羁', '羟', '羡', '翘', '翙', '翚', '耧', '耸', '耻', '聂',
            '聋', '职', '聍', '联', '聩', '聪', '肃', '肠', '肤', '肮', '肴', '肾', '肿', '胀',
            '胁', '胆', '胧', '胨', '胪', '胫', '胶', '脉', '脍', '脏', '脐', '脑', '脓', '脔',
            '脚', '脸', '腭', '腻', '腼', '腾', '膑', '舆', '舣', '舰', '舱', '舻', '艰', '艳',
            '艺', '节', '芈', '芗', '芜', '芦', '苁', '苇', '苈', '苋', '苌', '苍', '苎', '苏',
            '茎', '茏', '茑', '茔', '茕', '茧', '荆', '荙', '荚', '荛', '荜', '荞', '荟', '荠',
            '荡', '荣', '荤', '荥', '荦', '荧', '荨', '荩', '荪', '荫', '荬', '荭', '药', '莅',
            '莱', '莲', '莳', '莴', '莶', '获', '莸', '莹', '莺', '莼', '萝', '萤', '营', '萦',
            '萧', '萨', '蒉', '蒋', '蒌', '蓝', '蓟', '蓣', '蓥', '蓦', '蔷', '蔹', '蔺', '蔼',
            '蕰', '蕲', '蕴', '薮', '藓', '虏', '虑', '虚', '虬', '虱', '虽', '虾', '虿', '蚀',
            '蚁', '蚂', '蚕', '蚬', '蛊', '蛎', '蛏', '蛮', '蛰', '蛱', '蛲', '蛳', '蛴', '蜗',
            '蝇', '蝈', '蝉', '蝼', '蝾', '螀', '螨', '衅', '衔', '补', '衬', '衮', '袄', '袅',
            '袆', '袜', '袭', '装', '裆', '裈', '裢', '裤', '褛', '褴', '襕', '见', '观', '规',
            '觅', '视', '觇', '览', '觉', '觊', '觋', '觍', '觎', '觏', '觐', '觑', '觞', '触',
            '觯', '誉', '誊', '计', '订', '讣', '认', '讥', '讦', '讧', '讨', '让', '讪', '讫',
            '训', '议', '讯', '记', '讲', '讳', '讴', '讵', '讶', '讷', '许', '讹', '论', '讼',
            '讽', '设', '访', '诀', '证', '诂', '诃', '评', '诅', '识', '诇', '诈', '诉', '诊',
            '诋', '词', '诎', '诏', '译', '诒', '诓', '诔', '试', '诖', '诗', '诘', '诙', '诚',
            '诛', '诜', '话', '诞', '诟', '诠', '诡', '询', '诣', '诤', '该', '详', '诧', '诨',
            '诩', '诫', '诬', '语', '诮', '误', '诰', '诱', '诲', '诳', '说', '诵', '诶', '请',
            '诸', '诹', '诺', '读', '诽', '课', '诿', '谀', '谁', '谂', '调', '谄', '谅', '谆',
            '谈', '谊', '谋', '谌', '谍', '谎', '谏', '谐', '谑', '谒', '谓', '谔', '谕', '谗',
            '谘', '谙', '谚', '谛', '谜', '谟', '谠', '谡', '谢', '谣', '谤', '谥', '谦', '谧',
            '谨', '谩', '谪', '谬', '谭', '谮', '谯', '谰', '谱', '谲', '谳', '谴', '谵', '谶',
            '贝', '贞', '负', '贡', '财', '责', '贤', '败', '账', '货', '质', '贩', '贪', '贫',
            '贬', '购', '贮', '贯', '贰', '贱', '贲', '贳', '贴', '贵', '贶', '贷', '贸', '费',
            '贺', '贻', '贼', '贽', '贾', '贿', '赁', '赂', '赃', '资', '赅', '赇', '赈', '赉',
            '赊', '赋', '赌', '赍', '赎', '赏', '赐', '赑', '赒', '赓', '赔', '赕', '赖', '赘',
            '赚', '赛', '赜', '赝', '赞', '赟', '赠', '赡', '赢', '赣', '赪', '赵', '赶', '趋',
            '趸', '跃', '跄', '跞', '践', '跶', '跷', '跸', '跻', '踌', '踪', '踯', '蹑', '蹒',
            '蹰', '蹿', '躏', '躯', '车', '轧', '轨', '轩', '轪', '轫', '转', '轭', '轮', '软',
            '轰', '轱', '轲', '轳', '轴', '轵', '轶', '轸', '轹', '轻', '轼', '载', '轾', '轿',
            '辂', '较', '辄', '辅', '辆', '辇', '辈', '辉', '辊', '辋', '辍', '辎', '辏', '辐',
            '辑', '输', '辔', '辕', '辖', '辗', '辘', '辙', '辞', '辩', '辫', '边', '辽', '达',
            '迁', '过', '迈', '运', '还', '这', '进', '远', '违', '连', '迟', '迩', '迹', '选',
            '逊', '递', '逻', '遗', '遥', '邓', '邝', '邬', '邮', '邹', '邺', '邻', '郏', '郐',
            '郑', '郓', '郦', '郧', '郸', '酝', '酦', '酱', '酿', '释', '鉴', '銮', '钆', '钇',
            '针', '钉', '钊', '钋', '钌', '钍', '钎', '钏', '钐', '钑', '钒', '钓', '钔', '钕',
            '钖', '钗', '钘', '钙', '钚', '钛', '钝', '钞', '钟', '钠', '钡', '钢', '钣', '钤',
            '钥', '钦', '钧', '钨', '钩', '钪', '钫', '钬', '钭', '钮', '钯', '钰', '钱', '钲',
            '钳', '钴', '钵', '钶', '钷', '钸', '钹', '钺', '钻', '钼', '钽', '钾', '钿', '铀',
            '铁', '铂', '铃', '铄', '铅', '铆', '铈', '铉', '铊', '铋', '铌', '铍', '铎', '铏',
            '铐', '铑', '铒', '铕', '铖', '铗', '铙', '铚', '铛', '铜', '铝', '铟', '铠', '铡',
            '铢', '铣', '铤', '铥', '铦', '铧', '铨', '铩', '铪', '铫', '铬', '铭', '铮', '铯',
            '铰', '铱', '铲', '铳', '铵', '银', '铷', '铸', '铹', '铺', '铻', '铼', '铽', '链',
            '铿', '销', '锁', '锂', '锃', '锄', '锅', '锆', '锇', '锈', '锉', '锋', '锌', '锍',
            '锎', '锏', '锐', '锑', '锓', '锔', '锕', '锗', '锘', '错', '锚', '锛', '锜', '锝',
            '锞', '锟', '锠', '锡', '锢', '锣', '锤', '锥', '锦', '锧', '锨', '锫', '锬', '锭',
            '键', '锯', '锰', '锲', '锴', '锵', '锶', '锷', '锸', '锹', '锻', '锽', '锾', '锿',
            '镀', '镁', '镂', '镄', '镅', '镆', '镇', '镈', '镉', '镊', '镋', '镌', '镍', '镎',
            '镏', '镐', '镑', '镒', '镓', '镔', '镕', '镖', '镗', '镘', '镛', '镜', '镝', '镞',
            '镠', '镡', '镣', '镤', '镥', '镦', '镧', '镨', '镪', '镫', '镬', '镭', '镯', '镰',
            '镱', '镲', '镳', '镴', '镶', '长', '门', '闩', '闪', '闫', '闬', '闭', '问', '闯',
            '闰', '闱', '闲', '闳', '间', '闵', '闷', '闸', '闹', '闺', '闻', '闼', '闽', '闾',
            '闿', '阀', '阁', '阂', '阃', '阄', '阅', '阆', '阇', '阈', '阉', '阊', '阋', '阌',
            '阍', '阎', '阏', '阐', '阑', '阔', '阕', '阖', '阗', '阘', '阙', '阚', '队', '阳',
            '阴', '阵', '阶', '际', '陆', '陇', '陈', '陉', '陕', '陨', '险', '随', '隐', '隶',
            '隽', '难', '雇', '雏', '雳', '雾', '霁', '霉', '霭', '靓', '静', '靥', '鞑', '鞒',
            '韦', '韧', '韩', '韪', '韫', '韬', '韵', '页', '顶', '顷', '顸', '项', '顺', '须',
            '顼', '顽', '顾', '顿', '颀', '颁', '颂', '预', '颅', '领', '颇', '颈', '颉', '颊',
            '颋', '颌', '颍', '颎', '颏', '颐', '频', '颓', '颔', '颕', '颖', '颗', '题', '颙',
            '颚', '颛', '颜', '额', '颞', '颠', '颡', '颢', '颣', '颤', '颥', '颦', '颧', '风',
            '飑', '飒', '飓', '飕', '飘', '飙', '飞', '飨', '饥', '饦', '饨', '饪', '饫', '饬',
            '饭', '饮', '饯', '饰', '饱', '饲', '饴', '饵', '饶', '饷', '饸', '饹', '饺', '饼',
            '饽', '饿', '馁', '馂', '馃', '馄', '馅', '馆', '馈', '馊', '馋', '馍', '馎', '馏',
            '馐', '馑', '馒', '馓', '馔', '馕', '马', '驭', '驮', '驯', '驰', '驱', '驳', '驴',
            '驵', '驶', '驷', '驸', '驹', '驺', '驻', '驼', '驽', '驾', '驿', '骀', '骁', '骂',
            '骃', '骄', '骅', '骆', '骇', '骈', '骊', '骋', '验', '骍', '骎', '骏', '骐', '骑',
            '骓', '骕', '骖', '骗', '骘', '骙', '骚', '骜', '骝', '骞', '骟', '骠', '骡', '骢',
            '骤', '骥', '骧', '髅', '髋', '髌', '鬓', '魇', '魉', '鱼', '鱽', '鱾', '鱿', '鲀',
            '鲁', '鲂', '鲅', '鲆', '鲇', '鲈', '鲉', '鲊', '鲋', '鲌', '鲍', '鲎', '鲏', '鲐',
            '鲑', '鲔', '鲕', '鲗', '鲘', '鲙', '鲚', '鲛', '鲜', '鲞', '鲟', '鲠', '鲡', '鲢',
            '鲣', '鲤', '鲥', '鲦', '鲧', '鲨', '鲩', '鲪', '鲫', '鲬', '鲭', '鲮', '鲯', '鲱',
            '鲲', '鲳', '鲴', '鲵', '鲶', '鲷', '鲸', '鲹', '鲻', '鲼', '鲽', '鲾', '鲿', '鳀',
            '鳁', '鳂', '鳃', '鳄', '鳅', '鳆', '鳇', '鳉', '鳊', '鳌', '鳍', '鳎', '鳏', '鳐',
            '鳑', '鳒', '鳓', '鳔', '鳕', '鳖', '鳗', '鳘', '鳙', '鳛', '鳜', '鳝', '鳞', '鳟',
            '鳡', '鳢', '鳣', '鸟', '鸠', '鸡', '鸢', '鸣', '鸥', '鸦', '鸨', '鸩', '鸪', '鸫',
            '鸬', '鸭', '鸮', '鸯', '鸰', '鸱', '鸲', '鸳', '鸵', '鸶', '鸷', '鸸', '鸹', '鸺',
            '鸻', '鸽', '鸾', '鸿', '鹀', '鹁', '鹂', '鹃', '鹄', '鹅', '鹆', '鹇', '鹈', '鹉',
            '鹊', '鹋', '鹌', '鹍', '鹎', '鹏', '鹑', '鹕', '鹖', '鹗', '鹘', '鹙', '鹚', '鹛',
            '鹜', '鹝', '鹞', '鹟', '鹠', '鹡', '鹢', '鹣', '鹤', '鹥', '鹦', '鹧', '鹨', '鹩',
            '鹪', '鹫', '鹬', '鹭', '鹰', '鹱', '鹲', '鹳', '麦', '麸', '黄', '黉', '黩', '黾',
            '鼋', '鼌', '鼍', '鼹', '齐', '齑', '齿', '龂', '龃', '龄', '龅', '龆', '龇', '龈',
            '龉', '龊', '龋', '龌', '龙', '龚', '龛', '龟',
        ])
    };
}
