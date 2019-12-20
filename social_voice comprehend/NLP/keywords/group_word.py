from __future__ import division

import os
import pandas as pd

path = os.path.dirname(__file__)

# word grouping
group_words = {
        'good':['good', 'great', 'excellent', 'nice', 'awesome', 'best', 'not bad', 'amazing', 'perfect', 'ok', 'okay', 'wonderful','fantastic','fine', 'better','well', 'wxcellent'],       
        'love':['love', 'loving', 'like','loved', 'liked'],
        'difficult':['difficult', 'difficulty'],
        'convenience':['convenient', 'convienience', 'conveniet', 'efficient', 'effective', 'efficiency', 'convinient', 'conveinent', 'convinenet', 'handy'],
        'easy': ['easily','simple'],
        'fast': ['quick','faster','quickly'],
        'secure':['safe', 'secured' ],
        'smooth': ['stable',],
        'save': ['saves', 'saved'],
        'clear': ['fresh'],
        'app':['app!', 'aps', 'ap'],
        'cant':['cannot', 'couldnt'],
        'can':['could'],
        'doesnt':['dont','didnt'],
        'please':['pls', 'plz', 'plss', 'pleas', 'plea'],
        'ui':['interface', 'design', 'layout'],
        'user':['users'],
        'fingerprint':['finger print'],
        'would':['will'],
        'create':['creating', 'creates', 'created'],
        'come':['coming','comes', 'came'],
        'access':['acces'],
        'see':['sees','see', 'saw', 'seen'],
        'note':['notes', 'noted'],
        'use':['use', 'using', 'used', 'uses'],
        'rely':['relies', 'relied'],
        'remittance':['remittances'],
        'satisfy': ['satisfied', 'satisfying'],
        'improve':['improving','improved','improves'],
        'confuse': ['confused', 'confusing', 'confuses'],
        'rebate': ['rebates', 'rebating'],
        'pay': ['paid', 'paying', 'payment'],
        'minute':['min', 'mins', 'minutes'],
        'thing':['things'],
        'make': ['mades','made', 'made', 'making'],
        'pass':['pas'],
        'update':['updates', 'updated', 'updating']}

en_word_group = {
        'simplicity_or_convenience': ['complicate','confuse','conveniently','simply','simple','easier','convenience', 'easy', 'concise', 'difficult', 'complicated', 'inconvenient', 'inconvenience'],
        'time_consuming': ['consume','wait','timely','shortly','slowly','immediately','longer','endless','quicker', 'fast', 'faster' ,'fastest' 'speed','waste', 'long', 'slow', 'time', 'second', 'hour', 'minute'],
        'decvice_compatible': ['wei','hua','xiao','mi','samgsung','compatibility','iphonex','huawei','xiaomi','soni','apple','galaxy','ipad','htc','lg', 'nokia','samsung','device', 'iphone','phone','andriod','android', 'compatible', 'incompatible', 'os', 'ios', 'io'],
        'security':['secure', 'security','privacy', 'risk', 'safety'],
        'blank_wrong':['blank', 'wrong'],
        'biometric_identification': ['touch','touchid','recognition','recognize','recognise','authentication','biometric', 'fingerprint', 'face','faceid', 'facial', 'print', 'finger'],
        'OCR':['identify','authenticate','scanner','scan', 'scanning', 'scam', 'camera', 'picture', 'photo','id','hkid', 'identity', 'detect','detection','identification'],
        'QR_Code_method':['qr'],
        'clear':['clear', 'clean', 'unclear', 'clearly'],
        'language_display':['cn','chinese', 'english', 'language','eng'],
        'charge':['penalty','price','fee', 'charge', 'free', 'high', 'low', 'freely'],
        'size':['big','small', 'huge', 'large', 'little'],
        'app_performance': ['work', 'open', 'restart', 'terminate','kill','steadily','smooth', 'smoothly', 'bug', 'stable', 'server','system', 'error', 'crash','busy', 'stuck', 'buggy', 'steady', 'unstable'],
        'verification_otp':['verify','mail','messager','messenger','otp','msg','sms','sm', 'token', 'email', 'message', 'verification'],
        'area_use':['uk','globally','abroad', 'journey','travel','area', 'region','mainland','oversea', 'global', 'foreign', 'country'],
        'login_operation':['login', 'access', 'logon'],
        'loading':['load','reload'],
        'upload_operation':['upload','file','document'],
        'install_operation':['download', 'install', 'instal','reinstall', 'uninstall'],
        'update':['update', 'upgrade'],
        'banking':['bank', 'banking'],
        'transaction_way':['send','transfer','pay', 'transaction', 'fps','payee'],
        'money_management':['wealth','withdraw','cashless','coin','cheque','save', 'saving','money', 'bill', 'cash', 'limit', 'atm', 'branch', 'deposit', 'wallet'],
        'loan_management':['loan', 'debt', 'debit', 'morgage'],
        'fund_or_insurance_management':['fund', 'mpf', 'insurance'],
        'expense':['expense', 'budget'],
        'fx_service':['fx', 'forex', 'exchange', 'currency', 'currencies', 'hkd', 'dollar'],
        'rate_offer':['rate','interest'],
        'account_registration':['account', 'accout' 'register', 'registration'],
        'card_management':['card', 'credit'],
        'interface':['display','beautifully','ui','layout', 'interface', 'ugly'],
        'customer_service':['service', 'agent', 'hotline','answer', 'reply', 'chat', 'telephone'],
        'password_management':['password', 'pw', 'pin', 'passcode'],
        'function':['function', 'functional', 'feature', 'functionality'],
        'icon_operation':['key','click', 'page','screenshot', 'screen', 'button', 'keyboard', 'icon', 'capture'],
        'experience':['experience', 'ux'],
        'investment_service':['stock', 'trade', 'trading', 'invest', 'investment'],
        'internet_connection':['wifi','online', 'net', 'web', 'website', 'internet', 'disconnect','connect','connection', 'netword'],
        'notification': ['notice','notify','notification', 'note'],
        'promotion':['promotional','reward', 'promotion', 'coupon', 'discount', 'benefit'],
        'recordation':['record', 'history', 'logging', 'log', 'logs'],
        'other_feature':['confirmation', 'menu', 'friendly', 'unfriendly', 'address'],
        'advertisement':['ad', 'advertising', 'advertisement'],
        'shopping_service':['toabao', 'shop', 'shopping', 'alibaba', 'hktvmall'],
        'instruction':['instruction', 'statement', 'guide', 'guideline','reference'],
        }

zh_word_group = {
#        '不安全':['危险','被盗刷','盗刷','被盗','微信不安全','不安全','隐患','说不安全','脸不安全','不安全好','越来越不安全','安全隐患很大','安全隐患'],
#        '安全':['安全','用卡安全','安全性','银行卡安全','安全系统','数据安全','信息安全','用手机安全','安全仪器','安全感','公共安全'],
#        '方便':['利落','便利性','效率高','挺便民','方便','便民','便利','舒适便捷','越来越方便','方便快捷','顺着方便','确实方便','使用方便','方便简单','方便老年人','脸方便'],
#        '二维码':['二维码本质','二维码','用二维码','好好用二维码','二维码起码','微信用同一个二维码','当初二维码','扫码','收款码'],
#        '微信/支付宝':['支付宝微信红包','用支付宝微信','微信','用微信支付宝','总觉微信银行卡不','手机微信','着现金对着微信','从来不用微信','微信手续费收','用微信','现金能存微信','微信不安全','支付宝微信','支付宝微信说','微信用同一个二维码','支付宝微信红包','用支付宝微信','支付宝五福红包','支付宝','用微信支付宝','用支付宝','支付宝余额宝','给支付宝','信息支付宝','马云支付宝','支付宝微信','垃圾支付宝给','支付宝微信说'],
#        '人脸识别':['人脸','人脸识别系统','嘴脸','人脸识别','脸','脸部','比对好人脸','脸感觉','人脸时代','脸收款机','刷脸','脸不安全','拍脸','大脸一扫而空','脸不','脸互刷','脸藏','脸不成功','用脸','脸长','双胞胎脸','大众脸','脸部信息','没脸','人脸信息','人脸技术不','脸上','脸未必','正脸','脸排队','脸不值钱','大脸','方刷脸','脸硬是','啥时候被盗脸','脸始终','脸收款功能','脸方便','脸技术'],
#        '双胞胎':['实际上双胞胎','双胞胎妹妹','双胞胎脸','双胞胎无法区分','好奇双胞胎','卵双胞胎'],  
#        '现金':['现金','卡无现金','现金卡','着现金对着微信','现金能存微信','现金支付','没人用现金','传统现金支付','用现金'],
#        '银行':['银行账户','银行','银行卡','银行早','中国银行早','银行无卡','中小银行','银行渠道','张银行卡','无人银行','银行员工','手机银行','银行唯一','商业银行','总觉微信银行卡不','工商银行合作','中国银行','银行存款','顾客银行大额存款','银行打官司','银行未来','银行卡安全','银行卡密码'],        
#        '银行卡':['银行卡','张银行卡','总觉微信银行卡不','银行卡安全','银行卡密码'],
#        '指纹识别':['指纹','指纹识别','指纹识别优势','指纹独一无二','比指纹','用指纹','指纹不更好','感觉指纹'],
#        '手续费':['免费','微信手续费收','老百姓高额手续费','手续费比利息','收费'],
#        '老人':['老一辈','老年人难度','老大爷','老子','衰老','老人家','方便老年人'],
#        '技术':['科技','技术','技术成熟再来','科学技术','技术远远','化妆技术','人脸技术不','这项技术','技术成熟','脸技术'],
#        '隐私':['隐私','个人隐私','隐私信息','个人隐私越来越不','隐私权','公民生物隐私'],
#        '风险':['风险','现阶段风险','风险大'],
#        '密码':['密码','检验密码','数字密码','银行卡密码','密码高高举'],
#        '地区':['国家','全国','地方', '外国','欧美','美国','国外','韩国','法国','发达国家'],
#        '福利':['圈子小福利', '信用卡福利', '优惠'],
#        '人性化体验':['消费习惯'],
#        '速度快':['速度','真快','快速','快'],
#        '太好了,不错':['太好了', '喜欢','真香','好','不错','赞','牛'],
        

        'update|update|update问题':['升級', 'update','更新'],
        '體驗好|體驗|體驗差':['體驗','friendly','中意','喜欢','開心','好用','離譜','不好','唔好','奇怪','差','唔錯','不錯','吾錯','贊','幾正','最正','正啦','好正'],
        '方便|方便|唔方便':['順利','穩定','流暢','不便','方便','即時','快捷','簡潔'],
        '客户服務好|客户服務|客户服務差':['客户服務','服務中心','服務員','服務台','服務'],
        '业务服务功能好|业务服务功能|业务服务功能问题':['功能','查詢服務','資訊服務','基金服務','匯率服務功能','賬户服務','服務百','服務喇','充值服務','網上服務','投資服務','銀行服務','服務速度快速增長','互聯網服務','銀行服務','定位服務','服務功能','迫人用電子銀行服務','其他服務','信用卡服務','服務費'],
        '優惠多/好|優惠|優惠少/差':['禮券','著數','優惠', '着數'],
        '好容易|容易|難/麻煩':['困難','煩','難','容易','好易'],
        '登記成功|登記|登記问题':['開户', '登記'],
        '空白畫面|空白畫面|空白畫面问题':['空白'],
        '彈app|彈app|彈app问题':['閃退','彈', '閃屏'],
        '系統|系統|系統问题/繁忙':['系統'],
        '報錯|報錯|報錯':['異常','error','錯誤','報錯','故障'],
        '不費時|時間耗费|費時':['小時','分鐘', '時間','費時'],
        '快|速度|慢':['耐', '慢','快捷','效率','即時', '幾快', '很快', '快速','速度','特快','極快','快點','快搞', '很久'],
        '資料|資料|資料问题':['信息','資料'],
        '介面靓|介面|介面醜':['畫面','首頁','界面','介面','主頁', '頁面', '設計','版面','靓', '醜', '老土'],
        '版本|版本|版本问题':['新版','舊版', '版本'],
        '登入|登入|登入问题':['登入', 'login'],
        '設備兼容性好|設備兼容性|設備兼容性问题':['wei','hua','xiao','mi','samgsung','compatibility','iphonex','huawei','xiaomi','soni','apple','galaxy','ipad','htc','lg', 'nokia','samsung','device', 'iphone','phone','andriod','android', 'compatible', 'incompatible', 'os', 'ios', 'io', '新手機','新機','兼容','小米','apple','iphone', 'xs', 'xsmax', 'ios', 'max', '华为'],
        '簡單|簡單|複雜':['簡潔','簡單','清晰','複雜'],
        '轉帳|轉帳|轉帳问题':['過數','比錢','滙錢','過錢','交錢','匯款','轉'],
        '身份證|身份證|身份證问题':['身分證','身份證'],
        '電郵|電郵|電郵问题':['郵'],
        '電話號碼|電話號碼|電話號碼问题':['電話號碼'],
        '安全|安全性|不安全':['安全','風險'],
        '網絡|網絡|網絡差':['網絡','數據', '流量'],
        '網頁|網頁|網頁问题':['網頁'],
        '網上服務|網上服務|網上服務问题':['網上'],
        '借貸|借貸|借貸问题':['貸'],
        '網上購物|網上購物|網上購物问题':['淘寶'],
        '印花/積分_多|印花/積分|印花/積分_少':['儲分','積分','印花'],
        '廣告|廣告|廣告问题':['廣告'],
        '結餘|結餘|結餘问题':['餘額','結餘'],
        '八達通|八達通|八達通问题':['八達通'],
        '流動保安編碼/密碼/驗證碼|流動保安編碼/密碼/驗證碼|流動保安編碼/密碼/驗證碼问题':['編碼','流動','保安','號碼','密碼', '驗證碼'],
        '慳錢|慳錢|唔慳錢':['慳'],
        '語言選擇|語言選擇|語言選擇问题':['英文', '中文', '語言'],
        '銀行卡|銀行卡|銀行卡问题':['銀行卡'],
        '交易|交易|交易问题':['交易'],
        '交卡數|交卡數|交卡數问题':['卡數'],
        '帳户|帳户|帳户问题':['賬户', '帳户','户口'],
        '收款|收款|收款问题':['收款','收錢'],
        '熱線|熱線|熱線问题':['熱線'],
        '重裝|重裝|重裝问题':['重裝'],
        'crash|crash|crash问题':['load', '卡住', '卡死', 'crash', '死機', 'hang'],
        '現金|現金|現金问题':['現金'],
        '支票|支票|支票问题':['支票'],
        '信用卡|信用卡|信用卡问题':['信用卡'],
        '儲蓄卡|儲蓄卡|儲蓄卡问题':['儲蓄卡', '雙幣卡','提款卡'],
        '銀聯卡|銀聯卡|銀聯卡问题':['銀聯卡'],
        '無卡/無現金|無卡/無現金|無卡/無現金问题':['無卡', '無現金'],
        '清晰|清晰|唔清晰':['清晰'],
        '按鈕|按鈕|按鈕问题':['按鈕', '選項', '按鍵','錯鍵'],
        '結單|結單|結單问题':['結單'],
        '消費|消費|消費问题':['消費'],
        '連結|連結|連結问题':['連結',],
        '認證方式|認證方式|認證方式问题':['認證', '指紋','識別','人面', '指模','face', '生物','fingerprint'],
        '增值|增值|增值问题':['增值','充值'],
        '資訊|資訊|資訊问题':['資訊'],
        '限額|限額|限額问题':['限額'],
        '費用|費用|費用问题':['手續費','費用'],
        '短訊|短訊|短訊问题':['短訊','短信', 'sms'],
        '紀錄|紀錄|紀錄问题':['紀錄','記錄'],
        '貨幣|貨幣|貨幣问题':['貨幣','外幣','港幣','匯率','人民幣'],
        '地址|地址|地址问题':['地址'],
        '利息好|利息|利息差':['利息','利率'],
        '投資理財功能好|投資理財|投資理財问题':['保險','投資','基金','證券','股票', '理財', '估價','牛熊','債券'],
        '儲蓄|儲蓄|儲蓄':['儲蓄','儲存'],
        '態度好|態度|態度差':['態度','禮貌','客氣','誠意'],
        '字體|字體|字體问题':['字體','字型','簡體','繁體'],
        '不骗人|骗人|骗人':['骗','屈客錢','屈錢','呃','厄'],
        '私隱|私隱|私隱问题':['私隱','建議私人訊息', '私人', '私人資料', '私人電話號碼','個人資料'],
        '技術好|技術|技術差':['技術'],
}  

# %%
def group_W(word):
    flag = 0
    for key, mylist in group_words.items():
        if word in mylist:
            flag = 1
            return key
            break
    if flag == 0:
        return word

corpus_list_word = list(pd.read_csv(os.path.join(path,'corpus_list_word.txt'),header=None)[0].str.strip())
def get_change(comment):
    # input a string and output a string, part of stem
    skip_set = ['updates', 'updated', 'updating', 'hing', 'ding', 'wed', 'sing', 'red', 'ios', 'pls','pleas', 'plea','ap','aps','dbs','fps', 'sms','pass','pas','pleased','mades','made', 'made', 'making', 'relies','min', 'thing', 'things', 'cos', 'mins', 'thing', 'minutes','satisfied','paying','rebates','rebating','improving','improved', 'improves','confused', 'confusing', 'confuses', 'created', 'relied','noted','remittance', 'use', 'using', 'used', 'uses', 'notes','sees','seeing', 'saves', 'acces', 'coming','comes','creating', 'creates', 'mas', 'banking', 'using', 'useless', 'always', 'access', 'news','dls', 'funding', 'fds']
    comment = comment.split(' ')
    change = []
    for w in comment:
        flag = 0
        if len(w)>=3 and w not in skip_set:
            if w[-3:] == 'ing':
                if w[:-3] in corpus_list_word:
                    w = w[:-3]
                    flag = 1
            elif w[-2:] == 'es':
                if w[:-2] in corpus_list_word:
                    w = w[:-2]
                    flag = 1
            elif w[-1:] == 's':
                if w[:-1] in corpus_list_word:
                    w = w[:-1]
                    flag = 1
            elif w[-2:] == 'ed':
                if w[:-2] in corpus_list_word:
                    w = w[:-2]
                    flag = 1
        if flag == 0 and group_W(w) != '':
            w = group_W(w)
        change.append(w)
    return ' '.join(change).strip()

# %%
def group_features_en(keyword, polarity = 'neutral'):
    word_label = []
    detail_word_label = []
    flag = 0
    for x in keyword.split(' '):
        for key, value in en_word_group.items():
            if x in value and polarity == 'negative':
                word_label.append('bad_'+key)
                detail_word_label.append('bad_'+key+':'+keyword)
                flag = 1
                
                break
            elif x in value and polarity == 'positive':
                word_label.append('good_'+key)
                detail_word_label.append('good_'+key+':'+keyword)
                flag = 1
                break
    if flag == 0:
        word_label.append('others')
        detail_word_label.append('others'+':'+keyword)
    return list(set(word_label)),list(set(detail_word_label))


#def get_groups(s):
#    return group_features(s)[0]
#def get_groups_detail(s):
#    return group_features(s)[1]

# %%
# chinese grouping
def group_features_zh(keyword, polarity = 'neutral'):
    word_label = []
    detail_word_label = []
    flag = 0
    for key, value in zh_word_group.items():
        for v in value:
            if v in keyword and polarity == 'negative':
                word_label.append(key.split('|')[-1])
                detail_word_label.append(key.split('|')[-1]+':'+keyword)
                flag = 1
                break
            elif v in keyword and polarity == 'positive':
                word_label.append(key.split('|')[0])
                detail_word_label.append(key.split('|')[0]+':'+keyword)
                flag = 1
                break
            elif v in keyword and polarity == 'neutral':
                word_label.append(key.split('|')[1])
                detail_word_label.append(key.split('|')[1]+':'+keyword)
                flag = 1
    if flag == 0:
        word_label.append(keyword)
        detail_word_label.append('others'+':'+keyword)
    return list(set(word_label)),list(set(detail_word_label))  