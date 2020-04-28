# How to run

1. Convert the raw data and store the result in `./train` and `./test`:
    
    convert xml to csv 
    use bs4 BeautifulSoup (parser.py)

```
python3.6 convert_raw_data.py
```

1. Train three models

model architectures (model_edu.py, model_trans.py, model_rlat.py) are placed in model_dir/ and the model checkpoint will be placed in saved_model/

```
python3.6 main.py --make_dataset --train_edu --train_trans --train_rlat
```

2. Then you can test your model performance as follow:

```
python3.6 test.py 
```
--gold_edu      : using golden edu
--test_micro    : performances reported in micro F1 score
--test_macro    : performances reported in micro F1 score
--convert_multi : covert binary tree to mutliway tree

3. To predict, run

```
python3.6 demo.py --input_file $1 --output_file $2 
```
where $1 specifies input text $2 specifies output json file

e.g.

input_file
    Chinese raw text (utf-8) (simplified)

    据统计，这些城市去年完成国内生产总值一百九十多亿元，比开放前的一九九一年增长九成多。国务院于一九九二年先后批准了黑河、凭祥、珲春、伊宁、瑞丽等十四个边境城市为对外开放城市，同时还批准这些城市设立十四个边境经济合作区。三年多来，这些城市社会经济发展迅速，地方经济实力明显增强；经济年平均增长百分之十七，高于全国年平均增长速度。

output: 

    n6
    ├── n0
    │   ├── s0 据统计，这些城市去年完成国内生产总值一百九十多亿元，
    │   └── s1 比开放前的一九九一年增长九成多。
    └── n5
        ├── n1
        │   ├── s2 国务院于一九九二年先后批准了黑河、凭祥、珲春、伊宁、瑞丽等十四个边境城市为对外开放城市，
        │   └── s3 同时还批准这些城市设立十四个边境经济合作区。
        └── n4
            ├── n2
            │   ├── s4 三年多来，这些城市社会经济发展迅速，
            │   └── s5 地方经济实力明显增强；
            └── n3
                ├── s6 经济年平均增长百分之十七，
                └── s7 高于全国年平均增长速度。

and also output a json format output_file 

    {
    'EDUs':[(edu1), (edu2)...(edun)]
    'tree':{'args':[(subtree)], 'sense':(sense), 'center':(center)}
    'relations':[{'arg1':(arg1),'arg2':(arg2),'sense':(sense),'center':(center)},{...},{...}]
    }
