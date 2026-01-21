## 概述

根据小鼠在视频中的运动轨迹，自动识别其社会性与非社会性行为



隐藏测试集中将包含大约 200 个视频。





## 评估 

基于lab划分，然后基于动作算F1，求平均值。然后计算所有lab的后求平均值为最终得分。

* mabe-f-beta是评估代码
* 每个实验室是独立的，因此存在单个实验室偏差样本会导致分数更低





## EDA

### lab_id

对应实验室id

```
{
 'CRIM13',
 'CalMS21_supplemental',
 'CalMS21_task1',
 'CalMS21_task2',
 'MABe22_keypoints',
 'MABe22_movies',
 
 'AdaptableSnail',
 'BoisterousParrot',
 'CautiousGiraffe',
 'DeliriousFly',
 'ElegantMink',
 'GroovyShrew',
 'InvincibleJellyfish',
 'JovialSwallow',
 'LyricalHare',
 'NiftyGoldfinch',
 'PleasantMeerkat',
 'ReflectiveManatee',
 'SparklingTapir',
 'TranquilPanther',
 'UppityFerret'}
```

* CalMS21、CRIM13 和 MABe22 仅存在于训练集中，这些是往届的数据集，没有对应标签，当前方案没有使用

* 两个 MABe22 数据集是从一组实验中随机抽取的一分钟视频片段集合，这些实验连续数天每天 24小时记录小鼠的行为。这意味着小鼠会睡觉——所以其中一些视频片段相当平静！此外，小鼠还喜欢挤成一团睡觉，这会导致此时的姿态估计噪声较大。如果分析中排除这些片段，只需检查 CSV 文件中的“mouse1 condition”字段：如果显示“lights on”，则表示小鼠（夜行性动物）更有可能正在睡觉。或者，也可以查看追踪数据本身，并剔除没有运动的片段。





### behaviors_labeled

* 该`behaviors_labeled`字段包含在 test.csv 中，用于指定你需要预测/评分的操作，不在评估范围内的不得分

```
{'["\'mouse1\',\'mouse2\',\'sniff\'", "mouse1\',\'mouse2\',\'attack\'", "mouse1\',\'mouse2\',\'dominance\'", "mouse1,mouse2,attack", "mouse1,mouse2,dominance", "mouse1,mouse2,sniff"]',
 '["\'mouse1\',\'mouse2\',\'sniff\'", "mouse1\',\'mouse2\',\'attack\'", "mouse1,mouse2,attack", "mouse1,mouse2,sniff"]',
 '["\'mouse1\',\'mouse2\',\'sniff\'", "mouse1\',\'mouse2\',\'attack\'", "mouse1,mouse2,sniff"]',
 '["mouse1,mouse2,allogroom", "mouse1,mouse2,attack", "mouse1,mouse2,dominancegroom", "mouse1,mouse2,escape", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital", "mouse1,self,dig", "mouse1,self,selfgroom", "mouse2,mouse1,sniffgenital"]',
 '["mouse1,mouse2,allogroom", "mouse1,mouse2,attemptmount", "mouse1,mouse2,intromit", "mouse1,mouse2,mount", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital", "mouse2,self,rear"]',
 '["mouse1,mouse2,allogroom", "mouse1,mouse2,ejaculate", "mouse1,mouse2,intromit", "mouse1,mouse2,mount", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital", "mouse2,self,rear", "mouse2,self,selfgroom"]',
 '["mouse1,mouse2,allogroom", "mouse1,mouse2,ejaculate", "mouse1,mouse2,intromit", "mouse1,mouse2,mount", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital", "mouse2,self,rear"]',
 '["mouse1,mouse2,allogroom", "mouse1,mouse2,intromit", "mouse1,mouse2,mount", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital", "mouse2,self,rear"]',
 '["mouse1,mouse2,allogroom", "mouse1,mouse2,mount", "mouse1,mouse2,sniff", "mouse2,self,rear"]',
 '["mouse1,mouse2,allogroom", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital", "mouse2,self,rear"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,attack", "mouse1,mouse2,attemptmount", "mouse1,mouse2,mount", "mouse1,mouse2,sniffbody", "mouse1,mouse2,sniffface", "mouse1,mouse2,sniffgenital"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,attack", "mouse1,mouse2,avoid", "mouse1,mouse2,chase", "mouse1,mouse2,chaseattack", "mouse1,mouse2,submit", "mouse1,mouse3,approach", "mouse1,mouse3,attack", "mouse1,mouse3,avoid", "mouse1,mouse3,chase", "mouse1,mouse3,chaseattack", "mouse1,mouse3,submit", "mouse1,mouse4,approach", "mouse1,mouse4,attack", "mouse1,mouse4,avoid", "mouse1,mouse4,chase", "mouse1,mouse4,chaseattack", "mouse1,mouse4,submit", "mouse1,self,rear", "mouse2,mouse1,approach", "mouse2,mouse1,attack", "mouse2,mouse1,avoid", "mouse2,mouse1,chase", "mouse2,mouse1,chaseattack", "mouse2,mouse1,submit", "mouse2,mouse3,approach", "mouse2,mouse3,attack", "mouse2,mouse3,avoid", "mouse2,mouse3,chase", "mouse2,mouse3,chaseattack", "mouse2,mouse3,submit", "mouse2,mouse4,approach", "mouse2,mouse4,attack", "mouse2,mouse4,avoid", "mouse2,mouse4,chase", "mouse2,mouse4,chaseattack", "mouse2,mouse4,submit", "mouse2,self,rear", "mouse3,mouse1,approach", "mouse3,mouse1,attack", "mouse3,mouse1,avoid", "mouse3,mouse1,chase", "mouse3,mouse1,chaseattack", "mouse3,mouse1,submit", "mouse3,mouse2,approach", "mouse3,mouse2,attack", "mouse3,mouse2,avoid", "mouse3,mouse2,chase", "mouse3,mouse2,chaseattack", "mouse3,mouse2,submit", "mouse3,mouse4,approach", "mouse3,mouse4,attack", "mouse3,mouse4,avoid", "mouse3,mouse4,chase", "mouse3,mouse4,chaseattack", "mouse3,mouse4,submit", "mouse3,self,rear", "mouse4,mouse1,approach", "mouse4,mouse1,attack", "mouse4,mouse1,avoid", "mouse4,mouse1,chase", "mouse4,mouse1,chaseattack", "mouse4,mouse1,submit", "mouse4,mouse2,approach", "mouse4,mouse2,attack", "mouse4,mouse2,avoid", "mouse4,mouse2,chase", "mouse4,mouse2,chaseattack", "mouse4,mouse2,submit", "mouse4,mouse3,approach", "mouse4,mouse3,attack", "mouse4,mouse3,avoid", "mouse4,mouse3,chase", "mouse4,mouse3,chaseattack", "mouse4,mouse3,submit", "mouse4,self,rear"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,attack", "mouse1,mouse2,avoid", "mouse1,mouse2,chase", "mouse1,mouse2,chaseattack", "mouse1,mouse2,submit", "mouse1,mouse3,approach", "mouse1,mouse3,attack", "mouse1,mouse3,avoid", "mouse1,mouse3,chase", "mouse1,mouse3,chaseattack", "mouse1,mouse3,submit", "mouse1,mouse4,avoid", "mouse1,mouse4,chase", "mouse1,self,rear", "mouse2,mouse1,approach", "mouse2,mouse1,attack", "mouse2,mouse1,avoid", "mouse2,mouse1,chase", "mouse2,mouse1,chaseattack", "mouse2,mouse1,submit", "mouse2,mouse3,approach", "mouse2,mouse3,attack", "mouse2,mouse3,avoid", "mouse2,mouse3,chase", "mouse2,mouse3,chaseattack", "mouse2,mouse3,submit", "mouse2,mouse4,avoid", "mouse2,mouse4,chase", "mouse2,self,rear", "mouse3,mouse1,approach", "mouse3,mouse1,attack", "mouse3,mouse1,avoid", "mouse3,mouse1,chase", "mouse3,mouse1,chaseattack", "mouse3,mouse1,submit", "mouse3,mouse2,approach", "mouse3,mouse2,attack", "mouse3,mouse2,avoid", "mouse3,mouse2,chase", "mouse3,mouse2,chaseattack", "mouse3,mouse2,submit", "mouse3,self,rear", "mouse4,mouse1,avoid", "mouse4,mouse2,avoid"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,attack", "mouse1,mouse2,avoid", "mouse1,mouse2,chase", "mouse1,mouse2,chaseattack", "mouse1,mouse2,submit", "mouse1,mouse3,approach", "mouse1,mouse3,attack", "mouse1,mouse3,avoid", "mouse1,mouse3,chase", "mouse1,mouse3,chaseattack", "mouse1,mouse3,submit", "mouse1,mouse4,avoid", "mouse1,self,rear", "mouse2,mouse1,approach", "mouse2,mouse1,attack", "mouse2,mouse1,avoid", "mouse2,mouse1,chase", "mouse2,mouse1,chaseattack", "mouse2,mouse1,submit", "mouse2,mouse3,approach", "mouse2,mouse3,attack", "mouse2,mouse3,avoid", "mouse2,mouse3,chase", "mouse2,mouse3,chaseattack", "mouse2,mouse3,submit", "mouse2,mouse4,attack", "mouse2,mouse4,avoid", "mouse2,mouse4,chase", "mouse2,mouse4,chaseattack", "mouse2,self,rear", "mouse3,mouse1,approach", "mouse3,mouse1,attack", "mouse3,mouse1,avoid", "mouse3,mouse1,chase", "mouse3,mouse1,chaseattack", "mouse3,mouse1,submit", "mouse3,mouse2,approach", "mouse3,mouse2,attack", "mouse3,mouse2,avoid", "mouse3,mouse2,chase", "mouse3,mouse2,chaseattack", "mouse3,mouse2,submit", "mouse3,self,rear", "mouse4,mouse1,avoid", "mouse4,mouse2,avoid"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,attack", "mouse1,mouse2,avoid", "mouse1,mouse2,chase", "mouse1,mouse2,chaseattack", "mouse1,mouse2,submit", "mouse1,mouse3,approach", "mouse1,mouse3,attack", "mouse1,mouse3,avoid", "mouse1,mouse3,chase", "mouse1,mouse3,chaseattack", "mouse1,mouse3,submit", "mouse1,self,rear", "mouse2,mouse1,approach", "mouse2,mouse1,attack", "mouse2,mouse1,avoid", "mouse2,mouse1,chase", "mouse2,mouse1,chaseattack", "mouse2,mouse1,submit", "mouse2,mouse3,approach", "mouse2,mouse3,attack", "mouse2,mouse3,avoid", "mouse2,mouse3,chase", "mouse2,mouse3,chaseattack", "mouse2,mouse3,submit", "mouse2,mouse4,attack", "mouse2,mouse4,avoid", "mouse2,mouse4,chase", "mouse2,mouse4,chaseattack", "mouse2,self,rear", "mouse3,mouse1,approach", "mouse3,mouse1,attack", "mouse3,mouse1,avoid", "mouse3,mouse1,chase", "mouse3,mouse1,chaseattack", "mouse3,mouse1,submit", "mouse3,mouse2,approach", "mouse3,mouse2,attack", "mouse3,mouse2,avoid", "mouse3,mouse2,chase", "mouse3,mouse2,chaseattack", "mouse3,mouse2,submit", "mouse3,self,rear", "mouse4,mouse2,attack"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,attack", "mouse1,mouse2,avoid", "mouse1,mouse2,chase", "mouse1,mouse2,chaseattack", "mouse1,mouse2,submit", "mouse1,mouse3,approach", "mouse1,mouse3,attack", "mouse1,mouse3,avoid", "mouse1,mouse3,chase", "mouse1,mouse3,chaseattack", "mouse1,mouse3,submit", "mouse1,self,rear", "mouse2,mouse1,approach", "mouse2,mouse1,attack", "mouse2,mouse1,avoid", "mouse2,mouse1,chase", "mouse2,mouse1,chaseattack", "mouse2,mouse1,submit", "mouse2,mouse3,approach", "mouse2,mouse3,attack", "mouse2,mouse3,avoid", "mouse2,mouse3,chase", "mouse2,mouse3,chaseattack", "mouse2,mouse3,submit", "mouse2,self,rear", "mouse3,mouse1,approach", "mouse3,mouse1,attack", "mouse3,mouse1,avoid", "mouse3,mouse1,chase", "mouse3,mouse1,chaseattack", "mouse3,mouse1,submit", "mouse3,mouse2,approach", "mouse3,mouse2,attack", "mouse3,mouse2,avoid", "mouse3,mouse2,chase", "mouse3,mouse2,chaseattack", "mouse3,mouse2,submit", "mouse3,self,rear"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,attack", "mouse1,mouse2,chase", "mouse1,mouse2,defend", "mouse1,mouse2,escape", "mouse1,mouse2,flinch", "mouse1,mouse2,follow", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffface", "mouse1,mouse2,sniffgenital", "mouse1,mouse2,tussle", "mouse1,self,biteobject", "mouse1,self,climb", "mouse1,self,dig", "mouse1,self,exploreobject", "mouse1,self,rear", "mouse1,self,selfgroom", "mouse2,mouse1,approach", "mouse2,mouse1,attack", "mouse2,mouse1,chase", "mouse2,mouse1,defend", "mouse2,mouse1,escape", "mouse2,mouse1,flinch", "mouse2,mouse1,follow", "mouse2,mouse1,sniff", "mouse2,mouse1,sniffface", "mouse2,mouse1,sniffgenital", "mouse2,mouse1,tussle", "mouse2,self,biteobject", "mouse2,self,climb", "mouse2,self,dig", "mouse2,self,exploreobject", "mouse2,self,rear", "mouse2,self,selfgroom"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,attack", "mouse1,mouse2,disengage", "mouse1,mouse2,mount", "mouse1,mouse2,sniff", "mouse1,self,rear", "mouse1,self,selfgroom"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,attack", "mouse1,mouse2,dominancemount", "mouse1,mouse2,sniffbody", "mouse1,mouse2,sniffface", "mouse1,mouse2,sniffgenital"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,attack", "mouse1,mouse2,sniff", "mouse2,mouse1,defend", "mouse2,self,freeze", "mouse2,self,rear"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,attack", "mouse1,mouse2,sniff", "mouse2,mouse1,defend", "mouse2,self,freeze"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,attack", "mouse1,mouse2,sniffbody", "mouse1,mouse2,sniffface", "mouse1,mouse2,sniffgenital"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,attack", "mouse1,mouse2,sniffbody", "mouse1,mouse2,sniffface"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,attack", "mouse1,mouse2,sniffbody", "mouse1,mouse2,sniffgenital"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,attack", "mouse1,mouse2,sniffbody"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,attack", "mouse2,mouse1,defend", "mouse2,mouse1,escape"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,attemptmount", "mouse1,mouse2,escape", "mouse1,mouse2,intromit", "mouse1,mouse2,mount", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital", "mouse1,self,dig", "mouse1,self,rear", "mouse1,self,rest", "mouse1,self,run", "mouse1,self,selfgroom", "mouse2,mouse1,approach", "mouse2,mouse1,escape", "mouse2,mouse1,sniff", "mouse2,mouse1,sniffgenital", "mouse2,self,dig", "mouse2,self,rear", "mouse2,self,rest", "mouse2,self,selfgroom"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,attemptmount", "mouse1,mouse2,escape", "mouse1,mouse2,intromit", "mouse1,mouse2,mount", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital", "mouse1,self,dig", "mouse1,self,rear", "mouse1,self,selfgroom", "mouse2,mouse1,approach", "mouse2,mouse1,escape", "mouse2,mouse1,sniff", "mouse2,mouse1,sniffgenital", "mouse2,self,dig", "mouse2,self,rear", "mouse2,self,selfgroom"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,attemptmount", "mouse1,mouse2,intromit", "mouse1,mouse2,mount", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffbody", "mouse1,mouse2,sniffface", "mouse1,mouse2,sniffgenital"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,attemptmount", "mouse1,mouse2,intromit", "mouse1,mouse2,mount", "mouse1,mouse2,sniffbody", "mouse1,mouse2,sniffface", "mouse1,mouse2,sniffgenital"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,attemptmount", "mouse1,mouse2,mount", "mouse1,mouse2,sniffbody", "mouse1,mouse2,sniffface", "mouse1,mouse2,sniffgenital"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,attemptmount", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital", "mouse1,self,climb", "mouse1,self,dig", "mouse1,self,rear", "mouse1,self,selfgroom", "mouse2,mouse1,approach", "mouse2,mouse1,escape", "mouse2,mouse1,sniff", "mouse2,mouse1,sniffgenital", "mouse2,self,climb", "mouse2,self,dig", "mouse2,self,rear", "mouse2,self,rest", "mouse2,self,selfgroom"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,chase", "mouse1,mouse2,escape", "mouse1,mouse2,reciprocalsniff", "mouse1,mouse2,sniffbody", "mouse1,mouse2,sniffgenital", "mouse2,mouse1,reciprocalsniff", "mouse2,mouse1,sniff"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,defend", "mouse1,mouse2,escape", "mouse1,mouse2,intromit", "mouse1,mouse2,mount", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital", "mouse1,self,dig", "mouse1,self,rear", "mouse1,self,rest", "mouse1,self,run", "mouse1,self,selfgroom", "mouse2,mouse1,approach", "mouse2,mouse1,defend", "mouse2,mouse1,escape", "mouse2,mouse1,sniff", "mouse2,self,rear", "mouse2,self,rest", "mouse2,self,run", "mouse2,self,selfgroom"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,defend", "mouse1,mouse2,escape", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital", "mouse1,self,climb", "mouse1,self,dig", "mouse1,self,rear", "mouse1,self,run", "mouse1,self,selfgroom", "mouse2,mouse1,approach", "mouse2,mouse1,escape", "mouse2,mouse1,sniff", "mouse2,self,dig", "mouse2,self,rear", "mouse2,self,rest", "mouse2,self,run", "mouse2,self,selfgroom"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,defend", "mouse1,mouse2,escape", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital", "mouse1,self,dig", "mouse1,self,rear", "mouse1,self,run", "mouse1,self,selfgroom", "mouse2,mouse1,approach", "mouse2,mouse1,sniff", "mouse2,mouse1,sniffgenital", "mouse2,self,climb", "mouse2,self,dig", "mouse2,self,rear", "mouse2,self,run", "mouse2,self,selfgroom"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,defend", "mouse1,mouse2,escape", "mouse1,mouse2,sniff", "mouse1,self,dig", "mouse1,self,rear", "mouse1,self,rest", "mouse1,self,run", "mouse1,self,selfgroom", "mouse2,mouse1,approach", "mouse2,mouse1,escape", "mouse2,mouse1,sniff", "mouse2,mouse1,sniffgenital", "mouse2,self,dig", "mouse2,self,rear", "mouse2,self,rest", "mouse2,self,run", "mouse2,self,selfgroom"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,defend", "mouse1,mouse2,escape", "mouse1,mouse2,sniff", "mouse1,self,dig", "mouse1,self,rear", "mouse1,self,rest", "mouse1,self,run", "mouse1,self,selfgroom", "mouse2,mouse1,approach", "mouse2,mouse1,escape", "mouse2,mouse1,sniff", "mouse2,self,dig", "mouse2,self,rear", "mouse2,self,rest", "mouse2,self,run", "mouse2,self,selfgroom"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,escape", "mouse1,mouse2,intromit", "mouse1,mouse2,mount", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital", "mouse1,self,dig", "mouse1,self,rear", "mouse1,self,rest", "mouse1,self,selfgroom", "mouse2,mouse1,approach", "mouse2,mouse1,escape", "mouse2,mouse1,sniff", "mouse2,mouse1,sniffgenital", "mouse2,self,climb", "mouse2,self,dig", "mouse2,self,rear", "mouse2,self,rest", "mouse2,self,run", "mouse2,self,selfgroom"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,escape", "mouse1,mouse2,intromit", "mouse1,mouse2,mount", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital", "mouse1,self,dig", "mouse1,self,rear", "mouse1,self,run", "mouse1,self,selfgroom", "mouse2,mouse1,approach", "mouse2,mouse1,escape", "mouse2,mouse1,sniff", "mouse2,mouse1,sniffgenital", "mouse2,self,climb", "mouse2,self,dig", "mouse2,self,rear", "mouse2,self,rest", "mouse2,self,run", "mouse2,self,selfgroom"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,escape", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital", "mouse1,self,climb", "mouse1,self,dig", "mouse1,self,rear", "mouse1,self,selfgroom", "mouse2,mouse1,approach", "mouse2,mouse1,escape", "mouse2,mouse1,sniff", "mouse2,mouse1,sniffgenital", "mouse2,self,dig", "mouse2,self,rear", "mouse2,self,selfgroom"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,escape", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital", "mouse1,self,dig", "mouse1,self,rear", "mouse1,self,rest", "mouse1,self,run", "mouse1,self,selfgroom", "mouse2,mouse1,approach", "mouse2,mouse1,defend", "mouse2,mouse1,escape", "mouse2,mouse1,sniff", "mouse2,mouse1,sniffgenital", "mouse2,self,climb", "mouse2,self,dig", "mouse2,self,rear", "mouse2,self,rest", "mouse2,self,run", "mouse2,self,selfgroom"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,escape", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital", "mouse1,self,dig", "mouse1,self,rear", "mouse1,self,selfgroom", "mouse2,mouse1,approach", "mouse2,mouse1,escape", "mouse2,mouse1,sniff", "mouse2,mouse1,sniffgenital", "mouse2,self,dig", "mouse2,self,rear", "mouse2,self,selfgroom"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,intromit", "mouse1,mouse2,mount", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital", "mouse1,self,rear", "mouse1,self,rest", "mouse1,self,selfgroom", "mouse2,mouse1,escape", "mouse2,mouse1,sniff", "mouse2,self,rear", "mouse2,self,rest", "mouse2,self,selfgroom"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,intromit", "mouse1,mouse2,mount", "mouse1,mouse2,sniffbody", "mouse1,mouse2,sniffface", "mouse1,mouse2,sniffgenital", "mouse1,self,genitalgroom"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,intromit", "mouse1,mouse2,mount", "mouse1,mouse2,sniffbody", "mouse1,mouse2,sniffface", "mouse1,mouse2,sniffgenital"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,intromit", "mouse1,mouse2,mount", "mouse1,mouse2,sniffbody", "mouse1,mouse2,sniffgenital"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,mount", "mouse1,mouse2,sniffbody", "mouse1,mouse2,sniffface", "mouse1,mouse2,sniffgenital", "mouse1,self,genitalgroom"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,mount", "mouse1,mouse2,sniffbody", "mouse1,mouse2,sniffface", "mouse1,mouse2,sniffgenital"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,mount", "mouse1,mouse2,sniffbody", "mouse1,mouse2,sniffface", "mouse1,self,genitalgroom"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital", "mouse1,self,climb", "mouse1,self,dig", "mouse1,self,rear", "mouse1,self,selfgroom", "mouse2,mouse1,approach", "mouse2,mouse1,escape", "mouse2,mouse1,sniff", "mouse2,mouse1,sniffgenital", "mouse2,self,dig", "mouse2,self,rear", "mouse2,self,rest", "mouse2,self,run", "mouse2,self,selfgroom"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital", "mouse1,self,dig", "mouse1,self,rear", "mouse1,self,rest", "mouse1,self,run", "mouse1,self,selfgroom", "mouse2,mouse1,approach", "mouse2,mouse1,escape", "mouse2,mouse1,sniff", "mouse2,mouse1,sniffgenital", "mouse2,self,climb", "mouse2,self,dig", "mouse2,self,rear", "mouse2,self,run", "mouse2,self,selfgroom"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,sniffbody", "mouse1,mouse2,sniffface", "mouse1,mouse2,sniffgenital"]',
 '["mouse1,mouse2,approach", "mouse1,mouse2,sniffbody", "mouse1,mouse2,sniffface"]',
 '["mouse1,mouse2,attack", "mouse1,mouse2,chase", "mouse1,mouse2,escape", "mouse1,mouse2,follow", "mouse2,mouse1,attack", "mouse2,mouse1,chase", "mouse2,mouse1,escape", "mouse2,mouse1,follow"]',
 '["mouse1,mouse2,attack", "mouse1,mouse2,chase", "mouse1,mouse2,escape", "mouse2,mouse1,attack", "mouse2,mouse1,chase", "mouse2,mouse1,escape"]',
 '["mouse1,mouse2,attack", "mouse1,mouse2,chase", "mouse1,mouse2,sniff"]',
 '["mouse1,mouse2,attack", "mouse1,mouse2,dominancemount", "mouse1,mouse2,mount", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital"]',
 '["mouse1,mouse2,attack", "mouse1,mouse2,dominancemount", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital"]',
 '["mouse1,mouse2,attack", "mouse1,mouse2,dominancemount", "mouse1,mouse2,sniff"]',
 '["mouse1,mouse2,attack", "mouse1,mouse2,intromit", "mouse1,mouse2,mount", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital", "mouse2,self,rear", "mouse2,self,selfgroom"]',
 '["mouse1,mouse2,attack", "mouse1,mouse2,intromit", "mouse1,mouse2,mount", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital", "mouse2,self,rear"]',
 '["mouse1,mouse2,attack", "mouse1,mouse2,mount", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital"]',
 '["mouse1,mouse2,attack", "mouse1,mouse2,mount", "mouse1,mouse2,sniff"]',
 '["mouse1,mouse2,attack", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital"]',
 '["mouse1,mouse2,attack", "mouse1,mouse2,sniff"]',
 '["mouse1,mouse2,attack", "mouse1,mouse2,sniffbody", "mouse1,mouse2,sniffface", "mouse1,mouse2,sniffgenital"]',
 '["mouse1,mouse2,attack", "mouse2,mouse1,defend", "mouse2,mouse1,escape"]',
 '["mouse1,mouse2,attack", "mouse2,mouse1,escape"]',
 '["mouse1,mouse2,attack"]',
 '["mouse1,mouse2,attemptmount", "mouse1,mouse2,dominancemount", "mouse1,mouse2,mount", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital"]',
 '["mouse1,mouse2,attemptmount", "mouse1,mouse2,intromit", "mouse1,mouse2,mount", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital", "mouse2,self,rear", "mouse2,self,selfgroom"]',
 '["mouse1,mouse2,attemptmount", "mouse1,mouse2,intromit", "mouse1,mouse2,mount", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital", "mouse2,self,rear"]',
 '["mouse1,mouse2,attemptmount", "mouse1,mouse2,intromit", "mouse1,mouse2,mount", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital"]',
 '["mouse1,mouse2,attemptmount", "mouse1,mouse2,mount", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital"]',
 '["mouse1,mouse2,attemptmount", "mouse1,mouse2,mount", "mouse1,mouse2,sniff"]',
 '["mouse1,mouse2,dominancemount", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital"]',
 '["mouse1,mouse2,ejaculate", "mouse1,mouse2,intromit", "mouse1,mouse2,mount", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital", "mouse2,self,rear", "mouse2,self,selfgroom"]',
 '["mouse1,mouse2,intromit", "mouse1,mouse2,mount", "mouse1,mouse2,reciprocalsniff", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital", "mouse1,self,huddle", "mouse2,mouse1,reciprocalsniff", "mouse2,mouse1,sniffgenital", "mouse2,self,rear", "mouse2,self,selfgroom"]',
 '["mouse1,mouse2,intromit", "mouse1,mouse2,mount", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital", "mouse1,self,rear", "mouse1,self,rest", "mouse1,self,selfgroom", "mouse2,mouse1,approach", "mouse2,mouse1,defend", "mouse2,mouse1,escape", "mouse2,mouse1,sniff", "mouse2,mouse1,sniffgenital", "mouse2,self,climb", "mouse2,self,dig", "mouse2,self,rear", "mouse2,self,rest", "mouse2,self,selfgroom"]',
 '["mouse1,mouse2,intromit", "mouse1,mouse2,mount", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital", "mouse2,mouse1,sniffgenital", "mouse2,self,rear", "mouse2,self,selfgroom"]',
 '["mouse1,mouse2,intromit", "mouse1,mouse2,mount", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital", "mouse2,self,rear", "mouse2,self,selfgroom"]',
 '["mouse1,mouse2,intromit", "mouse1,mouse2,mount", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital", "mouse2,self,rear"]',
 '["mouse1,mouse2,intromit", "mouse1,mouse2,mount", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital"]',
 '["mouse1,mouse2,intromit", "mouse1,mouse2,mount", "mouse1,mouse2,sniff"]',
 '["mouse1,mouse2,mount", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffface", "mouse1,mouse2,sniffgenital"]',
 '["mouse1,mouse2,mount", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital"]',
 '["mouse1,mouse2,mount", "mouse1,mouse2,sniff", "mouse2,self,rear"]',
 '["mouse1,mouse2,mount", "mouse1,mouse2,sniff"]',
 '["mouse1,mouse2,mount", "mouse1,mouse2,sniffbody", "mouse1,mouse2,sniffface", "mouse1,mouse2,sniffgenital", "mouse1,self,genitalgroom"]',
 '["mouse1,mouse2,mount", "mouse1,mouse2,sniffbody", "mouse1,mouse2,sniffface", "mouse1,mouse2,sniffgenital"]',
 '["mouse1,mouse2,mount", "mouse2,mouse1,defend"]',
 '["mouse1,mouse2,mount"]',
 '["mouse1,mouse2,reciprocalsniff", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital", "mouse1,self,huddle", "mouse2,mouse1,reciprocalsniff", "mouse2,mouse1,sniffgenital", "mouse2,self,rear", "mouse2,self,selfgroom"]',
 '["mouse1,mouse2,reciprocalsniff", "mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital", "mouse1,self,huddle", "mouse2,mouse1,reciprocalsniff", "mouse2,mouse1,sniffgenital", "mouse2,self,rear"]',
 '["mouse1,mouse2,shepherd", "mouse2,mouse1,shepherd"]',
 '["mouse1,mouse2,sniff", "mouse1,mouse2,sniffface", "mouse1,mouse2,sniffgenital"]',
 '["mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital", "mouse2,mouse1,sniffgenital", "mouse2,self,rear"]',
 '["mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital", "mouse2,self,rear", "mouse2,self,selfgroom"]',
 '["mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital", "mouse2,self,rear"]',
 '["mouse1,mouse2,sniff", "mouse1,mouse2,sniffgenital"]',
 '["mouse1,mouse2,sniff", "mouse2,mouse1,defend", "mouse2,self,freeze", "mouse2,self,rear"]',
 '["mouse1,mouse2,sniff", "mouse2,mouse1,defend"]',
 '["mouse1,mouse2,sniff", "mouse2,self,rear"]',
 '["mouse1,mouse2,sniff"]',
 '["mouse1,mouse2,sniffbody", "mouse1,mouse2,sniffface", "mouse1,mouse2,sniffgenital"]',
 '["mouse1,mouse2,sniffgenital", "mouse2,mouse1,defend", "mouse2,mouse1,escape"]',
 '["mouse2,mouse1,defend", "mouse2,mouse1,escape"]',
 '["mouse2,mouse1,defend", "mouse2,self,freeze", "mouse2,self,rear"]',
 '["mouse2,mouse1,defend", "mouse2,self,freeze"]',
 nan}
```



### 动作

```
{"attack",
 'allogroom',
 'approach',
 'attemptmount',
 'avoid',
 'biteobject',
 'chase',
 'chaseattack',
 'climb',
 'defend',
 'dig',
 'disengage',
 'dominance',
 'dominancegroom',
 'dominancemount',
 'ejaculate',
 'escape',
 'exploreobject',
 'flinch',
 'follow',
 'freeze',
 'genitalgroom',
 'huddle',
 'intromit',
 'mount',
 'rear',
 'reciprocalsniff',
 'rest',
 'run',
 'self',
 'selfgroom',
 'shepherd',
 'sniff',
 'sniffbody',
 'sniffface',
 'sniffgenital',
 'submit',
 'tussle'}
```



```
self-action labels: {'biteobject', 'climb', 'dig', 'exploreobject', 'freeze', 'genitalgroom', 'huddle', 'rear', 'rest', 'run', 'selfgroom'} 

interaction labels: {'allogroom', 'approach', 'attack', 'attemptmount', 'avoid', 'chase', 'chaseattack', 'defend', 'disengage', 'dominance', 'dominancegroom', 'dominancemount', 'ejaculate', 'escape', 'flinch', 'follow', 'intromit', 'mount', 'reciprocalsniff', 'shepherd', 'sniff', 'sniffbody', 'sniffface', 
```





### body_parts_tracked

提供的身体部位数据，不同实验室追踪的不一样

```
{'["body_center", "ear_left", "ear_right", "forepaw_left", "forepaw_right", "hindpaw_left", "hindpaw_right", "neck", "nose", "tail_base", "tail_midpoint", "tail_tip"]',
 '["body_center", "ear_left", "ear_right", "headpiece_bottombackleft", "headpiece_bottombackright", "headpiece_bottomfrontleft", "headpiece_bottomfrontright", "headpiece_topbackleft", "headpiece_topbackright", "headpiece_topfrontleft", "headpiece_topfrontright", "lateral_left", "lateral_right", "neck", "nose", "tail_base", "tail_midpoint", "tail_tip"]',
 '["body_center", "ear_left", "ear_right", "hip_left", "hip_right", "lateral_left", "lateral_right", "nose", "spine_1", "spine_2", "tail_base", "tail_middle_1", "tail_middle_2", "tail_tip"]',
 '["body_center", "ear_left", "ear_right", "lateral_left", "lateral_right", "neck", "nose", "tail_base", "tail_midpoint", "tail_tip"]',
 '["body_center", "ear_left", "ear_right", "lateral_left", "lateral_right", "nose", "tail_base", "tail_tip"]',
 '["body_center", "ear_left", "ear_right", "lateral_left", "lateral_right", "nose", "tail_base"]',
 '["body_center", "ear_left", "ear_right", "nose", "tail_base"]',
 '["ear_left", "ear_right", "head", "tail_base"]',
 '["ear_left", "ear_right", "hip_left", "hip_right", "neck", "nose", "tail_base"]',
 '["ear_left", "ear_right", "nose", "tail_base", "tail_tip"]'}
```



```
body_parts_tracked: ["body_center", "ear_left", "ear_right", "forepaw_left", "forepaw_right", "hindpaw_left", "hindpaw_right", "neck", "nose", "tail_base", "tail_midpoint", "tail_tip"], train shape: (0, 38)
body_parts_tracked: ["body_center", "ear_left", "ear_right", "headpiece_bottombackleft", "headpiece_bottombackright", "headpiece_bottomfrontleft", "headpiece_bottomfrontright", "headpiece_topbackleft", "headpiece_topbackright", "headpiece_topfrontleft", "headpiece_topfrontright", "lateral_left", "lateral_right", "neck", "nose", "tail_base", "tail_midpoint", "tail_tip"], train shape: (7, 38)
body_parts_tracked: ["body_center", "ear_left", "ear_right", "hip_left", "hip_right", "lateral_left", "lateral_right", "nose", "spine_1", "spine_2", "tail_base", "tail_middle_1", "tail_middle_2", "tail_tip"], train shape: (21, 38)
body_parts_tracked: ["body_center", "ear_left", "ear_right", "lateral_left", "lateral_right", "neck", "nose", "tail_base", "tail_midpoint", "tail_tip"], train shape: (10, 38)
body_parts_tracked: ["body_center", "ear_left", "ear_right", "lateral_left", "lateral_right", "nose", "tail_base", "tail_tip"], train shape: (42, 38)
body_parts_tracked: ["body_center", "ear_left", "ear_right", "lateral_left", "lateral_right", "nose", "tail_base"], train shape: (74, 38)
body_parts_tracked: ["body_center", "ear_left", "ear_right", "nose", "tail_base"], train shape: (19, 38)
body_parts_tracked: ["ear_left", "ear_right", "head", "tail_base"], train shape: (17, 38)
body_parts_tracked: ["ear_left", "ear_right", "hip_left", "hip_right", "neck", "nose", "tail_base"], train shape: (634, 38)
body_parts_tracked: ["ear_left", "ear_right", "nose", "tail_base", "tail_tip"], train shape: (24, 38)

```

* 过滤后的数量





## 方案

整体流程

```
 → 数据清洗
 → 单鼠 / 双鼠特征工程
 → 子采样 + 树模型（LGBM / XGB / CatBoost）
 → 多模型集成
 → 自适应阈值 + 时序平滑
 → 行为片段输出（submission）
```

* 特征工程 + 树模型，手工构造 几何 / 速度 / 互动特征，数模型对离散的特征友好
* 解决不同视频帧率不一致问题，通过 _scale() 进行 FPS 归一化，
* 单鼠（Single Mouse）特征刻画“这个鼠在干什么”，几何结构特征，运动学特征（FPS-aware），状态与节律等
* 双鼠（Pair）特征刻画“鼠 A 对 鼠 B 在干什么”，跨鼠距离特征，接近 / 远离 / 追逐，协调与同步等
* StratifiedSubsetClassifier 分层采样以处理不平衡数据。基于动作训练模型，判断某个动作是否在某个时间发生

* 用 多弱相关模型 换稳定性和泛化，不同树深，learning rate，n_estimators，输出概率融合
* 预测后处理，行为级输出而非帧级，帧概率时间平滑（rolling mean），连续区间切段，过滤极短事件（<3 帧）





## 高分方案

* 选取多数实验室共有的 5 个关键身体部位，将不同 lab 数据合并建模，增强模型对分布差异的适应能力。
* 按 lab × action 进行阈值网格搜索，并通过概率/阈值比值解决多动作冲突，显著优化 F1 表现。

* 正/负/Masked 目标设计，精确处理不完整标注，避免错误监督；

* CNN + Transformer 架构，同时建模时间动态与双鼠/多部位关系；

* NN 侧使用 多尺度卷积 + 残差结构，结合 全实验室预训练 + 按 lab 微调，并通过 损失加权、gated max、后处理 logit 调整 强化稀有行为识别。

* 采用 GNN（TransformerConv）建模社会交互 + Squeezeformer 捕捉时序模式 + Pairwise / RelationNetwork 分类头，将“社会行为 + 个体自中心时序建模”有效结合。

* 过高 dropout（最高 0.35）、GNN/Transformer 交错、是否显式 edge features、RelationNetwork 头等方式构建多模型家族，为最终集成提供互补性。

* 用行为特定阈值（0.22–0.35）替代统一阈值

	
