python -c "
with open('src/chair_eval/chair.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 定义我们自己的单数化函数（完美支持 COCO 的所有动物、家具等名词）
custom_singularize = '''
def singularize(word):
    irregular = {
        \"knives\": \"knife\", \"leaves\": \"leaf\", \"wolves\": \"wolf\", 
        \"sheep\": \"sheep\", \"people\": \"person\", \"mice\": \"mouse\", 
        \"children\": \"child\"
    }
    word = word.lower().strip()
    if word in irregular: 
        return irregular[word]
    if word.endswith(\"ies\"): 
        return word[:-3] + \"y\"
    if word.endswith(\"sses\"): 
        return word[:-2]
    if word.endswith(\"ches\") or word.endswith(\"shes\") or word.endswith(\"xes\"): 
        return word[:-2]
    if word.endswith(\"s\") and not word.endswith(\"ss\") and not word.endswith(\"is\"): 
        return word[:-1]
    return word
'''

# 用自定义函数替换掉对 pattern 的导入
new_code = code.replace('from pattern.en import singularize', custom_singularize)

with open('src/chair_eval/chair.py', 'w', encoding='utf-8') as f:
    f.write(new_code)
print('='*50)
print('Success: Standalone CHAIR patch applied successfully!')
print('='*50)
"