from __future__ import annotations
import pandas as pd
from numpy import log2
from dataclasses import dataclass, field
from typing import Dict, Any, Union

## Read table

df = pd.read_excel('animels_data.xlsx')

## Drop "id" columns (name and serial number)

df = df.drop(['מספר', 'שם'], axis=1)

## Rename columns to english

df.rename(
    columns={
        'משפחה': 'family',
        'בעל רגליים': 'legs',
        'סביבת החיים במים': 'water',
        'בעל יכולת לעוף': 'fly',
        'לידה': 'birth',
    },
    inplace=True
)

df.replace(
    {
        'כן': 'Yes',
        'לא': 'No',
        'לפעמים': 'Sometimes',
        'יונקים': 'Yes',
        'זוחלים': 'No',
        'לפעמים': 'Sometimes',
    },
    inplace=True
)

MAX_DEPTH = 2

@dataclass
class TreeNode:
    # name of the attribute we checking or the group prediction
    name: str
    is_leaf: bool = False
    
    # The keys of "sons" are the answers for the attribute sored in "name"
    # The values are the sub-tree
    sons: Dict[Any, Union[dict, TreeNode]] = field(default_factory=lambda: {})
        
    # Depth of this node
    depth: int = 0

def entropia(df: pd.DataFrame) -> float:
    df_temp = df.copy()
    df_temp.loc[:,'cnt'] = df_temp['family'].copy()
    
    df_grp = df_temp[['family', 'cnt']].groupby(['family']).count()
    df_grp = df_grp.reset_index()
    
    total_cnt, _ = df.shape
    
    # calculate p_i
    df_grp['prob'] = df_grp.cnt.apply(lambda n: n/total_cnt)
    # calculate -p_i*log(p_i)
    df_grp['grp_value'] = df_grp.apply(lambda row: -row['prob'] * log2(row['prob']), axis=1)
    
    # sum all and return
    return df_grp.grp_value.sum()

def entropia_A(df: pd.DataFrame, attribute_col: str) -> float:
    total_cnt, _ = df.shape # |D|
    
    total_sum = 0
    for attr_val in df[attribute_col].drop_duplicates():
        df_attr = df[df[attribute_col] == attr_val] # D_j
        attr_cnt, _ = df_attr.shape # |D_j|
        attr_entropia = entropia(df_attr)
        
        total_sum += attr_cnt * attr_entropia
    
    return total_sum / total_cnt

def gain(df: pd.DataFrame, attribute_col: str) -> float:
    ent = entropia(df)
    ent_a = entropia_A(df, attribute_col)
    
    return ent - ent_a

def get_leaf(df: pd.DataFrame) -> TreeNode:
    df_temp = df.copy()
    df_temp.loc[:,'cnt'] = df_temp['family'].copy()

    df_grp = df_temp[['family', 'cnt']].groupby(['family']).count()
    
    predictions = df_grp[df_grp.cnt == df_grp.cnt.max()]
    # if there are two diffrent options with the same probability
    # the first one is taken
    prediction = predictions.index[0]

    leaf = TreeNode(prediction, is_leaf=True, depth=MAX_DEPTH)

    return leaf

def get_root(df: pd.DataFrame, depth: int, show=False) -> TreeNode:
    attributes = df.columns.drop('family')
    
    if depth == MAX_DEPTH or len(attributes) == 0:
        return get_leaf(df)
    
    # not max depth yet, find best gain
    best_gain, best_gainer = 0, ''
    
    for attr in attributes:
        this_gain = gain(df, attr)
        
        # Optional, show the diffrent gains
        if show:
            print(f'Gain of {attr}:\t\t {this_gain}')
        
        # update best attribute to choose
        if best_gain < this_gain:
            best_gain = this_gain
            best_gainer = attr
    
    
    if best_gainer:
        # create a node
        root_node = TreeNode(best_gainer, depth=depth)
        return root_node
    else:
        return get_leaf(df)

def add_sons(df: pd.DataFrame, root: TreeNode) -> TreeNode:
    if root.is_leaf:
        return root
    
    result = {}
    sons = df[root.name].unique()
    
    for s in sons:
        result[s] = df[df[root.name] == s].drop(root.name, axis=1)
    
    root.sons = result
    return root

def convert_sons_to_node(root: TreeNode) -> TreeNode:
    for son in root.sons:
        df_temp = root.sons[son]
        son_node = get_root(df_temp, depth=root.depth+1)
        son_node = add_sons(df_temp, son_node)

        root.sons[son] = son_node
    
    return root

def recursion_create_tree(root: TreeNode) -> TreeNode:
    if not root.is_leaf:
        root = convert_sons_to_node(root)
        for son in root.sons:
            root.sons[son] = recursion_create_tree(root.sons[son])
    
    return root

def ID3(df) -> TreeNode:
    root = get_root(df, depth=0)
    root = add_sons(df, root)
    return recursion_create_tree(root)

def tree_to_string(root: TreeNode):
    res = f'[{root.name}]\n'
    inde = '\t' * (root.depth+1)
    for son in root.sons:
        sub_tree_string = tree_to_string(root.sons[son])
        result_temp = f'{inde}({son})-->{sub_tree_string}\n'
        res += result_temp
    
    return res

root = ID3(df)

print(tree_to_string(root))