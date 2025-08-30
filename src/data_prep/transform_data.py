input_path = 'data/raw/heart_disease.csv'
output_path = 'data/processed/heart_disease.csv'

with open(input_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    line = line.replace('Yes', '1').replace('No', '0')
    line = line.replace('yes', '1').replace('no', '0')
    line = line.replace('Female', '0').replace('Male', '1')
    line = line.replace('Low', '0').replace('Medium', '2').replace('High', '3')
    new_lines.append(line)

with open(output_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f'Arquivo salvo em: {output_path}')