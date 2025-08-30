input_path = 'data/raw/heart_2020_cleaned.csv'
output_path = 'data/processed/heart_disease.csv'

with open(input_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    line = line.replace('Yes', '1').replace('No', '0')
    line = line.replace('yes', '1').replace('no', '0')
    line = line.replace('Female', '0').replace('Male', '1')
    line = line.replace('Low', '0').replace('Medium', '2').replace('High', '3')
    line = line.replace('18-24', '0').replace('25-29', '1').replace('30-34', '2').replace('35-39', '3').replace('40-44', '4').replace('45-49', '5').replace('50-54', '6').replace('55-59', '7').replace('60-64', '8').replace('65-69', '9').replace('70-74', '10').replace('75-79', '11').replace('80 or older', '12')
    line = line.replace('Excellent', '4').replace('Very good', '3').replace('Good', '2').replace('Fair', '1').replace('Poor', '0')
    line = line.replace('borderline diabetes', '2').replace('Borderline diabetes', '2')
    line = line.replace('borderline', '2').replace('Borderline', '2')
    line = line.replace('Yes (during pregnancy)', '3').replace('yes (during pregnancy)', '3')
    new_lines.append(line)

with open(output_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f'Arquivo salvo em: {output_path}')