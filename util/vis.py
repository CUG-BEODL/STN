import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

def plot_block_probabilities(num):

    slices = []
    scores_0= []
    scores_1 = []
    scores_2 = []

    with open('predict/demo.txt', 'r', encoding='utf-8') as file:
        #Flag
        block_started = False
        for line in file:
            if f"—— Block {num} Results ——" in line:
                block_started = True
                continue

            if block_started:
                if "—— Block" in line or "----------------------------------------" in line:
                    break

                if f"Predicted Class = " in line:
                    # 搜寻到切片索引
                    index_Slice_s = line.find('Slice ')
                    index_Slice_e = line.find(': Score0')
                    slice_num = int(line[index_Slice_s+len('Slice '):index_Slice_e])
                    # 搜寻到Score_0
                    index_s0_s = line.find('Score0 = ')
                    index_s0_e = line.find(', Score1')
                    score0 = float(line[index_s0_s+len('Score0 = '):index_s0_e])
                    # 搜寻到Score_1
                    index_s1_s = line.find('Score1 = ')
                    index_s1_e = line.find(', Score2')
                    score1 = float(line[index_s1_s+len('Score1 = '):index_s1_e])
                    # 搜寻到Score_2
                    index_s2_s = line.find('Score2 = ')
                    index_s2_e = line.find(', Predicted Class')
                    score2 = float(line[index_s2_s+len('Score2 = '):index_s2_e])
                    
                    slices.append(slice_num)
                    scores_0.append(score0)
                    scores_1.append(score1)
                    scores_2.append(score2)
    
    start_date = np.datetime64('2021-01-01')
    date_ticks = np.array([start_date.astype('datetime64[M]') + np.timedelta64(i, 'M') for i in slices])

    plt.figure(figsize=(10, 4))

    plt.plot(date_ticks, scores_0, marker='^', linestyle='-', color='#94c6cd', label='Probability of stable')

    plt.plot(date_ticks, scores_1, marker='s', linestyle='-', color='#E29135', label='Probability of expansion')

    plt.plot(date_ticks, scores_2, marker='o', linestyle='-', color='#72B063', label='Probability of demolition')

    plt.title(f'Demo picture')
    plt.xlabel('Date of each slice')
    plt.ylabel('Semantic change probability')
    plt.ylim(-0.05, 1.05)

    plt.xticks(date_ticks[::5], rotation=45)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./img/{num}.png', dpi=300)
    #plt.show()

#plot_block_probabilities(1022)