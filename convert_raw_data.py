from bs4 import BeautifulSoup
import pandas as pd
import os


def parse_xml( soup ):
    relation = soup.find_all("r") # find all relation
    ls = []

    first = True

    for r in relation:
        tmpd = {
            "p_id": r.find_parent('p').attrs['id'],
            "r_id": r.attrs['id'],
            "StructureType": r.attrs['structuretype'],
            "ConnectiveType": r.attrs['connectivetype'],
            "Layer": r.attrs['layer'],
            "RelationNumber": r.attrs['relationnumber'],
            "Connective": r.attrs['connective'],
            "RelationType": r.attrs['relationtype'],
            "ConnectivePosition": r.attrs['connectiveposition'],
            "ConnectiveAttribute": r.attrs['connectiveattribute'],
            "RoleLocation": r.attrs['rolelocation'],
            "LanguageSense": r.attrs['languagesense'],
            "Sentence": r.attrs['sentence'],
            "SentencePosition": r.attrs['sentenceposition'],
            "Center": r.attrs['center'],
            "ChildList": r.attrs['childlist'],
            "ParentId": r.attrs['parentid'],
            "UseTime": r.attrs['usetime'],
        }
        ls.append(tmpd)
    df = pd.DataFrame(ls, columns=["p_id", "r_id", "StructureType", "ConnectiveType", "Layer", "RelationNumber", "Connective", "RelationType", "ConnectivePosition", "ConnectiveAttribute", "RoleLocation", "LanguageSense", "Sentence", "SentencePosition", "Center", "ChildList", "ParentId", "UseTime"])

    return df


# file = ["raw_data-20190127T101957Z-001/raw_data/train_repair/001.xml"]

counter = 1
counter = '{:03d}'.format(counter)
# path = 'raw_data-20190127T101957Z-001/raw_data/train_repair/' + str(counter1) + '.xml'

for i in range(0,1000):

    path = 'raw_data-20190127T101957Z-001/raw_data/test_repair/' + str(counter) + '.xml'
    if(os.path.isfile(path)):
        soup = BeautifulSoup(open(path))
        df = parse_xml(soup)
        df.to_csv( "test/"+str(counter)+".csv", sep=",", index=False)

    counter = int(counter)+1
    counter = '{:03d}'.format(counter)