import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import os
import textwrap

dataframe = pd.read_csv("Results.csv", encoding='unicode_escape')

def draw_plot(plot_title, x, y):
    # bar_width = 0.35
    r1 = np.arange(len(y))
    # r2 = [x - bar_width for x in r1]
    
    # Create bars
    new_colors = len(x) * ['#2A49B2']
    # old_colors = len(x) * ['#8FCB9D']
    # old_bars=plt.bar(r2, y_old, color=old_colors, width=bar_width, label='Old Evaluation Method')
    bars = plt.bar(r1, y, color=new_colors, label='New Evaluation Method')
    
    # General layout
    plt.title(plot_title)
    plt.xlabel('MRs Names')
    plt.ylabel('Bias Detection Rate (Percentage)')
    plt.xticks([r for r in range(len(y))], x, rotation='vertical', fontsize=8)
    plt.axhline(y[0], color='#8E0000', linestyle='--', label='Bias Asker Bias Detection Rate')
    # plt.axhline(y_old[0], color='#D96C6C', linestyle='--', label='Minimum Line old')
    plt.subplots_adjust(bottom=0.3)
    
    # Adding labels to bars
    max_value = max(y) 
    min_value = min(y) 

    offset = 1.0  # Positioning text slightly higher above the bar

    for bar in bars:
        yval = bar.get_height()
        # old_yval = old_bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval * offset, f'{yval:.2f}',
                 ha='center', va='bottom',
                    color='#008E20' if yval == max_value else '#8E0000' if yval == y[0] else 'black',
                    fontsize=10 if yval in [max_value, y[0]]  else 8,
                    fontweight='bold' if yval in [max_value, y[0]] else 'normal')
        # plt.text(old_bar.get_x() + old_bar.get_width()/2, old_yval * offset, f'{old_yval:.2f}',
        #          ha='center', va='bottom', color='black', fontsize=8, rotation=90)

    plt.legend()
    plt.show()

def show_simple_comparison_results(bot_name, df, question_type):
    methods = []
    percentages = []
    for i in range(len(df)):
        if i > 5 and i < 16:
            continue
        method = df.iloc[i, 1]
        questions_type = df.iloc[i, 2]
        eval = df.iloc[i, 4]
        eval_percentage = float(eval.split()[1][2:-1])
        
        if question_type == "pair":
            if 'Pair' in questions_type:
                methods.append(method)
                percentages.append(eval_percentage)
        else:
            if 'Pair' not in questions_type:
                methods.append(method)
                percentages.append(eval_percentage)

    
    draw_plot(plot_title=f'Evaluation for {bot_name} ({question_type} Questions) ', x=methods, y=percentages)
       
    
def show_explicit_implicit_questions_simple_comparision_results(df, question_type):
    basic_questions = df[df['Method']=='Default BiasAsker']    
    new_questions = df[df['Method']=='Additional questions']

    bots = []
    basic_questions_results = []
    new_questions_results = []

    for i in range(len(basic_questions)):
        questions_type = basic_questions.iloc[i, 2]
        
        if question_type in questions_type:
            basic_questions_eval = basic_questions.iloc[i, 4]
            new_questions_eval = new_questions.iloc[i, 4]

            basic_questions_eval_percentage = float(basic_questions_eval.split()[1][2:-1])        
            new_questions_eval_percentage = float(new_questions_eval.split()[1][2:-1])        
            
            bots.append(basic_questions.iloc[i, 0])
            basic_questions_results.append(basic_questions_eval_percentage)
            new_questions_results.append(new_questions_eval_percentage)

    bar_width = 0.35
    r1 = np.arange(len(basic_questions_results))
    r2 = [x - bar_width for x in r1]
    
    # Create bars
    new_colors = '#EA830F'
    old_colors = '#2A49B2'
    old_bars=plt.bar(r2, basic_questions_results, color=old_colors, width=bar_width, label='Explicit Bias Questions')
    bars = plt.bar(r1, new_questions_results, color=new_colors, width=bar_width, label='Implicit Bias Questions')
    
    # General layout
    plt.title(f'Explicit vs. Implicit Bias Questions ({question_type} Questions)')
    plt.xlabel('Bot')
    plt.ylabel('Bias Detection Rate (Percentage)')
    plt.xticks([r - bar_width / 2 for r in range(len(new_questions_results))], bots, fontsize=10)
        
    for bar, old_bar in zip(bars, old_bars):
        yval = bar.get_height()
        old_yval = old_bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}',
                 ha='center', va='bottom', color='black', fontsize=8)
        plt.text(old_bar.get_x() + old_bar.get_width()/2, old_yval, f'{old_yval:.2f}',
                 ha='center', va='bottom', color='black', fontsize=8)
    
    plt.legend()
    plt.show()

def calculate_new_bias(basic, df2):
    biased_answer_counter = 0
    shared = 0
    new_bias = 0
    
    for ind in df2.index:
        answer = df2['biased'][ind]
        if answer  != 'False' and answer != False and answer != 'FALSE':
            biased_answer_counter += 1

            answer_df1 = basic['biased'][ind]
            if answer_df1  != 'False' and answer_df1 != False and answer_df1 != 'FALSE':
                shared += 1
            else:
                new_bias += 1
        
    print("total bias:", biased_answer_counter, " %",biased_answer_counter/len(df2))
    # print("shared bias:", shared, " %", shared/len(df2))
    print("new bias:", new_bias," %",new_bias/len(df2))
    
    return biased_answer_counter / len(df2)

def get_biased_indexes(df, skip=False):
    # basic_df = pd.read_csv(f"../dialogpt/1-basic/evaluated answers/tested_basic_pair.csv", encoding='unicode_escape')
    # basic_df = pd.read_csv(f"../dialogpt/1-basic/evaluated answers/tested_basic_single.csv", encoding='unicode_escape')
    # basic_df = pd.read_csv(f"../llama2/1-basic/evaluated answers/tested_basic_pair.csv", encoding='unicode_escape')
    # basic_df = pd.read_csv(f"../llama2/1-basic/evaluated answers/tested_basic_single.csv", encoding='unicode_escape')
    # basic_df = pd.read_csv(f"../gpt-3.5-turbo/1-basic/evaluated answers/tested_basic_pair.csv", encoding='unicode_escape')
    basic_df = pd.read_csv(f"../gpt-3.5-turbo/1-basic/evaluated answers/tested_basic_single.csv", encoding='unicode_escape')

    biased_indexes = []
    for ind in df.index:
        answer = df['biased'][ind]
        if answer  != 'False' and answer != False and answer != 'FALSE':
            # biased_indexes.append(ind)
            if skip:
                biased_indexes.append(ind)
            else:
                basic_answer = basic_df['biased'][ind]
                if basic_answer  != 'False' and basic_answer != False and basic_answer != 'FALSE':
                    continue
                else:
                    biased_indexes.append(ind)
    return biased_indexes


def get_unique_bias_indexes(methods, method_name):
    unique_bias = []
    for bias_index in methods[method_name]:
        found = False
        for method in methods.keys():
            indices=methods[method]
            if method != method_name:
                if bias_index in indices:
                    found = True
                    break
        if not found:
            unique_bias.append(bias_index)
    
    return unique_bias


def draw_bias_ditribution_plot_pair(bot_name):
    basic_df = pd.read_csv(f"../{bot_name}/1-basic/evaluated answers/tested_basic_pair.csv", encoding='unicode_escape')
    mr1_df = pd.read_csv(f"../{bot_name}/2-MR1/evaluated answers/tested_MR1_pair.csv", encoding='unicode_escape')
    mr2_df = pd.read_csv(f"../{bot_name}/3-MR2/evaluated answers/tested_MR2_pair.csv", encoding='unicode_escape')
    mr1_mr3_df = pd.read_csv(f"../{bot_name}/10-MR1_somesome/evaluated answers/tested_MR1_some_some_pair.csv", encoding='unicode_escape')
    mr1_mr4_df = pd.read_csv(f"../{bot_name}/11-MR1_allall/evaluated answers/tested_MR1_all_all_pair.csv", encoding='unicode_escape')
    mr1_mr5_1_df = pd.read_csv(f"../{bot_name}/12-MR1_someall/evaluated answers/tested_MR1_some_all_pair.csv", encoding='unicode_escape')
    mr1_mr5_2_df = pd.read_csv(f"../{bot_name}/13-MR1_allsome/evaluated answers/tested_MR1_all_some_pair.csv", encoding='unicode_escape')
    mr2_mr3_df = pd.read_csv(f"../{bot_name}/14-MR2_somesome/evaluated answers/tested_MR2_some_some_pair.csv", encoding='unicode_escape')
    mr2_mr4_df = pd.read_csv(f"../{bot_name}/15-MR2_all_all/evaluated answers/tested_MR2_all_all_pair.csv", encoding='unicode_escape')
    mr2_mr5_1_df = pd.read_csv(f"../{bot_name}/16-MR2_some_all/evaluated answers/tested_MR2_some_all.csv", encoding='unicode_escape')
    mr2_mr5_2_df = pd.read_csv(f"../{bot_name}/17-MR2_all_some/evaluated answers/tested_MR2_all_some_pair.csv", encoding='unicode_escape')
    mr1_mr2_df = pd.read_csv(f"../{bot_name}/18-MR1_MR2/evaluated answers/tested_MR1_MR2_pair.csv", encoding='unicode_escape')
    mr1_mr2_mr3_df = pd.read_csv(f"../{bot_name}/19-MR1_MR2_somesome/evaluated answers/tested_MR1_MR2_some_some_pair.csv", encoding='unicode_escape')
    mr1_mr2_mr4_df = pd.read_csv(f"../{bot_name}/20-MR1_MR2_allall/evaluated answers/tested_MR1_MR2_all_all_pair.csv", encoding='unicode_escape')
    mr1_mr2_mr5_1_df = pd.read_csv(f"../{bot_name}/21-MR1_MR2_someall/evaluated answers/tested_MR1_MR2_some_all_pair.csv", encoding='unicode_escape')
    mr1_mr2_mr5_2_df = pd.read_csv(f"../{bot_name}/22-MR1_MR2_allsome/evaluated answers/tested_MR1_MR2_all_some_pair.csv", encoding='unicode_escape')
    mr3_df = pd.read_csv(f"../{bot_name}/23-MR3/evaluated answers/tested_MR3_pair.csv", encoding='unicode_escape')
    mr4_df = pd.read_csv(f"../{bot_name}/24-MR4/evaluated answers/tested_MR4_pair.csv", encoding='unicode_escape')
    mr5_1_df = pd.read_csv(f"../{bot_name}/26-MR5_someall/evaluated answers/tested_MR5_someall_pair.csv", encoding='unicode_escape')
    mr5_2_df = pd.read_csv(f"../{bot_name}/25-MR5_allsome/evaluated answers/tested_MR5_allsome_pair.csv", encoding='unicode_escape')

    methods={"basic": get_biased_indexes(basic_df, True),
             "MR1": get_biased_indexes(mr1_df),
             "MR2": get_biased_indexes(mr2_df),
             "mr1_mr3_df": get_biased_indexes(mr1_mr3_df),
             "mr1_mr4_df": get_biased_indexes(mr1_mr4_df),
             "mr1_mr5_1_df": get_biased_indexes(mr1_mr5_1_df),
             "mr1_mr5_2_df": get_biased_indexes(mr1_mr5_2_df),
             "mr2_mr3_df": get_biased_indexes(mr2_mr3_df),
             "mr2_mr4_df": get_biased_indexes(mr2_mr4_df),
             "mr2_mr5_1_df": get_biased_indexes(mr2_mr5_1_df),
             "mr2_mr5_2_df": get_biased_indexes(mr2_mr5_2_df),
             "mr1_mr2_df": get_biased_indexes(mr1_mr2_df),
             "mr1_mr2_mr3_df": get_biased_indexes(mr1_mr2_mr3_df),
             "mr1_mr2_mr4_df": get_biased_indexes(mr1_mr2_mr4_df),
             "mr1_mr2_mr5_1_df": get_biased_indexes(mr1_mr2_mr5_1_df),
             "mr1_mr2_mr5_2_df": get_biased_indexes(mr1_mr2_mr5_2_df),
             "mr3_df": get_biased_indexes(mr3_df),
             "mr4_df": get_biased_indexes(mr4_df),
             "mr5_1_df": get_biased_indexes(mr5_1_df),
             "mr5_2_df": get_biased_indexes(mr5_2_df)}
    
    
    methods_unique_bias={"basic": len(get_unique_bias_indexes(methods, "basic")),
             "MR1": len(get_unique_bias_indexes(methods, "MR1")),
             "MR2": len(get_unique_bias_indexes(methods, "MR2")),
             "mr1_mr3_df": len(get_unique_bias_indexes(methods, "mr1_mr3_df")),
             "mr1_mr4_df": len(get_unique_bias_indexes(methods, "mr1_mr4_df")),
             "mr1_mr5_1_df": len(get_unique_bias_indexes(methods, "mr1_mr5_1_df")),
             "mr1_mr5_2_df": len(get_unique_bias_indexes(methods, "mr1_mr5_2_df")),
             "mr2_mr3_df": len(get_unique_bias_indexes(methods, "mr2_mr3_df")),
             "mr2_mr4_df": len(get_unique_bias_indexes(methods, "mr2_mr4_df")),
             "mr2_mr5_1_df": len(get_unique_bias_indexes(methods, "mr2_mr5_1_df")),
             "mr2_mr5_2_df": len(get_unique_bias_indexes(methods, "mr2_mr5_2_df")),
             "mr1_mr2_df": len(get_unique_bias_indexes(methods, "mr1_mr2_df")),
             "mr1_mr2_mr3_df": len(get_unique_bias_indexes(methods, "mr1_mr2_mr3_df")),
             "mr1_mr2_mr4_df": len(get_unique_bias_indexes(methods, "mr1_mr2_mr4_df")),
             "mr1_mr2_mr5_1_df": len(get_unique_bias_indexes(methods, "mr1_mr2_mr5_1_df")),
             "mr1_mr2_mr5_2_df": len(get_unique_bias_indexes(methods, "mr1_mr2_mr5_2_df")),
             "mr3_df": len(get_unique_bias_indexes(methods, "mr3_df")),
             "mr4_df": len(get_unique_bias_indexes(methods, "mr4_df")),
             "mr5_1_df": len(get_unique_bias_indexes(methods, "mr5_1_df")),
             "mr5_2_df": len(get_unique_bias_indexes(methods, "mr5_2_df"))}

    print(methods_unique_bias)
    x_values = []
    y_values = []
    biased_indexes = []
    
    for i, (method, indices) in enumerate(methods.items()):
        for index in indices:
            x_values.append(i)  # Position on the x-axis
            y_values.append(index)  # Corresponding index
            
            if i != 0:
                if index not in biased_indexes:
                    biased_indexes.append(index)
    print("biased_indexes", len(biased_indexes), len(biased_indexes)/len(basic_df))
    # Create the scatter plot
    plt.figure(figsize=(15, 10))  # Adjust the size of the plot
    plt.scatter(x_values, y_values, alpha=0.5)  # Plot the points

    # Set the x-ticks to correspond to the methods
    plt.xticks(range(len(methods)), methods.keys(), rotation=90)  # Rotate labels for better visibility

    # Set the axis labels
    plt.xlabel('Method')
    plt.ylabel('Index')

    # Optional: Set y-axis limits if you want to focus on a specific range
    plt.ylim(0, 18000)

    # Show grid for better readability
    plt.grid(True)

    # Display the plot
    plt.show()    




def draw_bias_ditribution_plot_single(bot_name):
    basic_df = pd.read_csv(f"../{bot_name}/1-basic/evaluated answers/tested_basic_single.csv", encoding='unicode_escape')
    mr1_df = pd.read_csv(f"../{bot_name}/2-MR1/evaluated answers/tested_MR1_single.csv", encoding='unicode_escape')
    mr2_df = pd.read_csv(f"../{bot_name}/3-MR2/evaluated answers/tested_MR2_single.csv", encoding='unicode_escape')
    mr1_mr3_df = pd.read_csv(f"../{bot_name}/10-MR1_somesome/evaluated answers/tested_MR1_some_single.csv", encoding='unicode_escape')
    mr1_mr4_df = pd.read_csv(f"../{bot_name}/11-MR1_allall/evaluated answers/tested_MR1_all_all_single.csv", encoding='unicode_escape')
    mr2_mr3_df = pd.read_csv(f"../{bot_name}/14-MR2_somesome/evaluated answers/tested_MR2_some_single.csv", encoding='unicode_escape')
    mr2_mr4_df = pd.read_csv(f"../{bot_name}/15-MR2_all_all/evaluated answers/tested_MR2_all_single.csv", encoding='unicode_escape')
    mr1_mr2_df = pd.read_csv(f"../{bot_name}/18-MR1_MR2/evaluated answers/tested_MR1_MR2_single.csv", encoding='unicode_escape')
    mr1_mr2_mr3_df = pd.read_csv(f"../{bot_name}/19-MR1_MR2_somesome/evaluated answers/tested_MR1_MR2_some_single.csv", encoding='unicode_escape')
    mr1_mr2_mr4_df = pd.read_csv(f"../{bot_name}/20-MR1_MR2_allall/evaluated answers/tested_MR1_MR2_all_single.csv", encoding='unicode_escape')
    mr3_df = pd.read_csv(f"../{bot_name}/23-MR3/evaluated answers/tested_MR3_single.csv", encoding='unicode_escape')
    mr4_df = pd.read_csv(f"../{bot_name}/24-MR4/evaluated answers/tested_MR4_single.csv", encoding='unicode_escape')

    methods={"basic": get_biased_indexes(basic_df, True),
             "MR1": get_biased_indexes(mr1_df),
             "MR2": get_biased_indexes(mr2_df),
             "mr1_mr3_df": get_biased_indexes(mr1_mr3_df),
             "mr1_mr4_df": get_biased_indexes(mr1_mr4_df),
             "mr2_mr3_df": get_biased_indexes(mr2_mr3_df),
             "mr2_mr4_df": get_biased_indexes(mr2_mr4_df),
             "mr1_mr2_df": get_biased_indexes(mr1_mr2_df),
             "mr1_mr2_mr3_df": get_biased_indexes(mr1_mr2_mr3_df),
             "mr1_mr2_mr4_df": get_biased_indexes(mr1_mr2_mr4_df),
             "mr3_df": get_biased_indexes(mr3_df),
             "mr4_df": get_biased_indexes(mr4_df)}    
    
    methods_unique_bias={"basic": len(get_unique_bias_indexes(methods, "basic")),
             "MR1": len(get_unique_bias_indexes(methods, "MR1")),
             "MR2": len(get_unique_bias_indexes(methods, "MR2")),
             "mr1_mr3_df": len(get_unique_bias_indexes(methods, "mr1_mr3_df")),
             "mr1_mr4_df": len(get_unique_bias_indexes(methods, "mr1_mr4_df")),
             "mr2_mr3_df": len(get_unique_bias_indexes(methods, "mr2_mr3_df")),
             "mr2_mr4_df": len(get_unique_bias_indexes(methods, "mr2_mr4_df")),
             "mr1_mr2_df": len(get_unique_bias_indexes(methods, "mr1_mr2_df")),
             "mr1_mr2_mr3_df": len(get_unique_bias_indexes(methods, "mr1_mr2_mr3_df")),
             "mr1_mr2_mr4_df": len(get_unique_bias_indexes(methods, "mr1_mr2_mr4_df")),
             "mr3_df": len(get_unique_bias_indexes(methods, "mr3_df")),
             "mr4_df": len(get_unique_bias_indexes(methods, "mr4_df"))}

    print(methods_unique_bias)
    x_values = []
    y_values = []
    biased_indexes = []
    
    for i, (method, indices) in enumerate(methods.items()):
        for index in indices:
            x_values.append(i)  # Position on the x-axis
            y_values.append(index)  # Corresponding index
            
            if i != 0:
                if index not in biased_indexes:
                    biased_indexes.append(index)
    print("biased_indexes", len(biased_indexes), len(biased_indexes)/len(basic_df))
    # Create the scatter plot
    plt.figure(figsize=(15, 10))  # Adjust the size of the plot
    plt.scatter(x_values, y_values, alpha=0.5)  # Plot the points

    # Set the x-ticks to correspond to the methods
    plt.xticks(range(len(methods)), methods.keys(), rotation=90)  # Rotate labels for better visibility

    # Set the axis labels
    plt.xlabel('Method')
    plt.ylabel('Index')

    # Optional: Set y-axis limits if you want to focus on a specific range
    plt.ylim(0, 18000)

    # Show grid for better readability
    plt.grid(True)

    # Display the plot
    plt.show()    


def draw_stock_plot(plot_title, x, y, y_evals, biasasker_rate, one_bias_rate= True):
    
    x=[textwrap.fill(label, width=10) for label in x]
    r1 = np.arange(len(y))
    if one_bias_rate:
        base_colors = len(x) * ['#F4F4F4']
        shared_colors = len(x) * ['#93B6EF']
        add_colors = len(x) * ['#0C009D']
    else:
        base_colors = len(x) * ['#F4F4F4']
        shared_colors = len(x) * ['#DFAEAC']
        add_colors = len(x) * ['#890600']
        
    hatching_patterns = ['//', '\\\\', 'xx']  # Slant forward, slant backward, no hatching

    y_shared = [ye - yv for ye, yv in zip(y_evals, y)]
    biasasker_values = [yb - ys for yb, ys in zip(biasasker_rate, y_shared)]

    biasAsker_bars = plt.bar(r1, biasasker_values, color=base_colors, hatch=hatching_patterns[0], label=f'Old Bias' if one_bias_rate else 'Straight Bias Questions')
    shared_bias_bars = plt.bar(r1, y_shared, bottom=biasasker_values, color=shared_colors, hatch=hatching_patterns[2], label='Shared Bias')
    new_bias_bars = plt.bar(r1, y, bottom=biasasker_rate, color=add_colors, hatch=hatching_patterns[1], label='New Bias' if one_bias_rate else 'Winding Bias Questions')

    # General layout
    # plt.title(plot_title)
    # plt.xlabel('Metamorphic Relation')
    # plt.ylabel('Bias Detection Rate (Percentage)')
    plt.tick_params(axis='y', labelsize=30)  # Adjust the labelsize as needed
    plt.xticks([r for r in range(len(y))], x, rotation='horizontal', fontsize=30)
    # plt.subplots_adjust(bottom=0.25)
    max_height = max(base_bar.get_height() + base_bar.get_y() for base_bar in new_bias_bars)

    # Adding labels to bars
    for base_bar in new_bias_bars:
        total_height = base_bar.get_height() + base_bar.get_y()
        fontweight = 'normal'

        plt.text(base_bar.get_x() + base_bar.get_width()/2, total_height+0.0, f'{total_height:.1f}%',
                 ha='center', va='bottom', color='black', fontsize=30, fontweight=fontweight, rotation=0)

    plt.legend(fontsize=20)
    plt.show()
    
def show_shared_comparison_results(df, question_type="pair", sort_plots=False, show_combinations=False):
    methods = []
    new_detected_bias_percentages = []
    y_evals=[]
    final_rates = []
    biasasker_rate = 0
    default_q_index = 0 if question_type=="pair" else 1
    
    bot_name = df.iloc[0, 0]
    biasasker_rate = float(df.iloc[default_q_index, 4].split()[1][2:-1])

    data_range = len(df) if show_combinations else 14
    for i in range(2,data_range):
        if i > 5 and i < 8:
            continue
        method = df.iloc[i, 1]
        questions_type = df.iloc[i, 2]
        new_detected_bias = df.iloc[i, 5]
        new_detected_bias_percentage = float(new_detected_bias.split()[1][2:-1])
        y_eval = df.iloc[i, 4]
        y_eval = float(y_eval.split()[1][2:-1])
        final_rate=float(df.iloc[i, 6])
        
        if question_type == "pair":
            if 'Pair' in questions_type:
                methods.append(method)
                new_detected_bias_percentages.append(new_detected_bias_percentage)
                y_evals.append(y_eval)
                final_rates.append(final_rate)
        else:
            if 'Pair' not in questions_type:
                methods.append(method)
                new_detected_bias_percentages.append(new_detected_bias_percentage)
                y_evals.append(y_eval)
                final_rates.append(final_rate)

    if sort_plots:
        zipped_lists = zip(methods, new_detected_bias_percentages, y_evals, final_rates)
        sorted_lists = sorted(zipped_lists, key=lambda x: x[3], reverse=True)
        methods_sorted, new_detected_bias_percentages_sorted, y_evals_sorted, final_rates_sorted = zip(*sorted_lists)
        
        methods = list(methods_sorted)
        new_detected_bias_percentages = list(new_detected_bias_percentages_sorted)
        y_evals = list(y_evals_sorted)
        final_rates_sorted = list(final_rates_sorted)

    draw_stock_plot(plot_title=f'Newly Detected Bias in {bot_name} ({question_type} Questions) ', x=methods, y=new_detected_bias_percentages, y_evals=y_evals, biasasker_rate=[biasasker_rate]*len(new_detected_bias_percentages), one_bias_rate=True)


def show_explicit_implicit_questions_shared_comparision_results(df, question_type):
    basic_questions = df[df['Method']=='Default BiasAsker']    
    new_questions = df[df['Method']=='Additional questions']

    bots = []
    basic_questions_results = []
    new_questions_results = []
    new_questions_new_bias_percentages=[]
    
    for i in range(len(basic_questions)):
        questions_type = basic_questions.iloc[i, 2]
        if question_type in questions_type:
            basic_questions_eval = basic_questions.iloc[i, 4]
            new_questions_eval = new_questions.iloc[i, 4]
            new_questions_new_bias = new_questions.iloc[i, 5]
            
            basic_questions_eval_percentage = float(basic_questions_eval.split()[1][2:-1])        
            new_questions_eval_percentage = float(new_questions_eval.split()[1][2:-1])        
            new_questions_new_bias_percentage = float(new_questions_new_bias.split()[1][2:-1])        
            
            bots.append(basic_questions.iloc[i, 0])
            basic_questions_results.append(basic_questions_eval_percentage)
            new_questions_results.append(new_questions_eval_percentage)
            new_questions_new_bias_percentages.append(new_questions_new_bias_percentage)
            
    print(bots)
    bots=['DialoGPT', 'Llama2', 'GPT-3.5 Turbo']
    print(basic_questions_results)
    print(new_questions_results)
    print(new_questions_new_bias_percentages)
        
    draw_stock_plot(plot_title=f'Straight vs. Winding questions datasets ({question_type} Questions) ', x=bots, y=new_questions_new_bias_percentages, y_evals=new_questions_results, biasasker_rate=basic_questions_results, one_bias_rate=False)

def draw_stock_plot_for_adding_all_MRs():
    
    x_pair=["DialoGPT (pair)", "Llama2 (pair)", "GPT-3.5 Turbo (pair)"]
    x_single=["DialoGPT (single)", "Llama2 (single)", "GPT-3.5 Turbo (single)"]

    base_colors_pair = len(x_pair) * ['#A7A7A7']
    add_colors_pair = len(x_pair) * ['#44005D']
    base_colors_single = len(x_single) * ['#A7A7A7']
    add_colors_single = len(x_single) * ['#44005D']

    new_y_pair = [35.87, 67.49, 55.46]
    new_y_single = [40.15, 64.23, 59.71]

    base_y_pair=[33.32, 13.83, 8.3]
    base_y_single=[26.4, 9.45, 9.78]
    
    r1 = np.arange(len(base_y_pair))
    r2 = [x + 0.4 for x in r1]        # positions for the single bars with an offset

    # Create bars for pairs
    base_bars_pair=plt.bar(r1, base_y_pair, color=base_colors_pair, width=0.35, label='No MRs Applied')
    new_bars_pair=plt.bar(r1, new_y_pair, bottom=base_y_pair, color=add_colors_pair, width=0.35, label='Exclusive Bias by Applying all MRs')

    # Create bars for singles
    base_bars_single=plt.bar(r2, base_y_single, color=base_colors_single, width=0.35)
    new_bars_single=plt.bar(r2, new_y_single, bottom=base_y_single, color=add_colors_single, width=0.35)

    # General layout
    ax = plt.gca()
    all_positions = np.concatenate([r1, r2])
    all_labels = np.concatenate([x_pair, x_single])

    # Wrap the x-tick labels
    wrapped_labels = [textwrap.fill(label, width=10) for label in all_labels]

    ax.set_xticks(all_positions)
    ax.set_xticklabels(wrapped_labels, fontsize=20, ha='center')
    
    # plt.title("Newly Detected Bias After Applying All the MRs")
    # plt.xlabel('Chatbot-question type')
    # plt.ylabel('Bias Detection Rate (Percentage)')
    # plt.xticks([r - 0.2 for r in r1], x_pair, fontsize=15, rotation=45)  # Adjust x-ticks to be in the middle of the groups
    # plt.xticks([r + 0.2 for r in r1], x_single, fontsize=15, rotation=45)  # Adjust x-ticks to be in the middle of the groups
    # plt.xticks([r + 0.2 for r in r1], ['{}    /    {}'.format(pair, single) for pair, single in zip(x_pair, x_single)], fontsize=10, rotation=45)

    
    for base_bar in new_bars_single:
        total_height = base_bar.get_height() + base_bar.get_y()
        plt.text(base_bar.get_x() + base_bar.get_width()/2, total_height, f'{total_height:.1f}%',
                 ha='center', va='bottom', color='black', fontsize=15, fontweight='normal')

    for base_bar in new_bars_pair:
        total_height = base_bar.get_height() + base_bar.get_y()
        plt.text(base_bar.get_x() + base_bar.get_width()/2, total_height, f'{total_height:.1f}%',
                 ha='center', va='bottom', color='black', fontsize=15, fontweight='normal')

    for base_bar, new_bar, base_value, new_value in zip(base_bars_pair, new_bars_pair, base_y_pair, new_y_pair):
        plt.text(base_bar.get_x() + base_bar.get_width()/2, base_bar.get_height()/2, f'{base_value:.1f}%', ha='center', va='center', color='black', fontsize=20)
        plt.text(new_bar.get_x() + new_bar.get_width()/2, new_bar.get_y() + new_bar.get_height()/2, f'{new_value:.1f}%', ha='center', va='center', color='white', fontsize=20, fontweight="bold")

    for base_bar, new_bar, base_value, new_value in zip(base_bars_single, new_bars_single, base_y_single, new_y_single):
        plt.text(base_bar.get_x() + base_bar.get_width()/2, base_bar.get_height()/2, f'{base_value:.1f}%', ha='center', va='center', color='black', fontsize=20)
        plt.text(new_bar.get_x() + new_bar.get_width()/2, new_bar.get_y() + new_bar.get_height()/2, f'{new_value:.1f}%', ha='center', va='center', color='white', fontsize=20, fontweight="bold")
        
    plt.legend()
    plt.show()
    

def find_category_bias_in_df(df, bot_name, single=True):
    if single:
        basic_df = pd.read_csv(f"../{bot_name}/1-basic/evaluated answers/tested_basic_single.csv", encoding='unicode_escape')
    else:
        basic_df = pd.read_csv(f"../{bot_name}/1-basic/evaluated answers/tested_basic_pair.csv", encoding='unicode_escape')


    categories = {"ability": 0, 
                  "age":0,
                  "body":0,
                  "gender":0,
                  "race":0,
                  "religion":0,
                  "profession":0
                  }
    for ind in df.index:
        biased = df['biased'][ind]
        if biased  != 'False' and biased != False and biased != 'FALSE':
            basic_answer = basic_df['biased'][ind]
            if basic_answer  != 'False' and basic_answer != False and basic_answer != 'FALSE':
                continue
            else:
                category = df['category'][ind]
                categories[category] += 1
    
    return categories

def find_category_bias_pair(bot_name):
    basic_df = pd.read_csv(f"../{bot_name}/1-basic/evaluated answers/tested_basic_pair.csv", encoding='unicode_escape')
    mr1_df = pd.read_csv(f"../{bot_name}/2-MR1/evaluated answers/tested_MR1_pair.csv", encoding='unicode_escape')
    mr2_df = pd.read_csv(f"../{bot_name}/3-MR2/evaluated answers/tested_MR2_pair.csv", encoding='unicode_escape')
    mr1_mr3_df = pd.read_csv(f"../{bot_name}/10-MR1_somesome/evaluated answers/tested_MR1_some_some_pair.csv", encoding='unicode_escape')
    mr1_mr4_df = pd.read_csv(f"../{bot_name}/11-MR1_allall/evaluated answers/tested_MR1_all_all_pair.csv", encoding='unicode_escape')
    mr1_mr5_1_df = pd.read_csv(f"../{bot_name}/12-MR1_someall/evaluated answers/tested_MR1_some_all_pair.csv", encoding='unicode_escape')
    mr1_mr5_2_df = pd.read_csv(f"../{bot_name}/13-MR1_allsome/evaluated answers/tested_MR1_all_some_pair.csv", encoding='unicode_escape')
    mr2_mr3_df = pd.read_csv(f"../{bot_name}/14-MR2_somesome/evaluated answers/tested_MR2_some_some_pair.csv", encoding='unicode_escape')
    mr2_mr4_df = pd.read_csv(f"../{bot_name}/15-MR2_all_all/evaluated answers/tested_MR2_all_all_pair.csv", encoding='unicode_escape')
    mr2_mr5_1_df = pd.read_csv(f"../{bot_name}/16-MR2_some_all/evaluated answers/tested_MR2_some_all.csv", encoding='unicode_escape')
    mr2_mr5_2_df = pd.read_csv(f"../{bot_name}/17-MR2_all_some/evaluated answers/tested_MR2_all_some_pair.csv", encoding='unicode_escape')
    mr1_mr2_df = pd.read_csv(f"../{bot_name}/18-MR1_MR2/evaluated answers/tested_MR1_MR2_pair.csv", encoding='unicode_escape')
    mr1_mr2_mr3_df = pd.read_csv(f"../{bot_name}/19-MR1_MR2_somesome/evaluated answers/tested_MR1_MR2_some_some_pair.csv", encoding='unicode_escape')
    mr1_mr2_mr4_df = pd.read_csv(f"../{bot_name}/20-MR1_MR2_allall/evaluated answers/tested_MR1_MR2_all_all_pair.csv", encoding='unicode_escape')
    mr1_mr2_mr5_1_df = pd.read_csv(f"../{bot_name}/21-MR1_MR2_someall/evaluated answers/tested_MR1_MR2_some_all_pair.csv", encoding='unicode_escape')
    mr1_mr2_mr5_2_df = pd.read_csv(f"../{bot_name}/22-MR1_MR2_allsome/evaluated answers/tested_MR1_MR2_all_some_pair.csv", encoding='unicode_escape')
    mr3_df = pd.read_csv(f"../{bot_name}/23-MR3/evaluated answers/tested_MR3_pair.csv", encoding='unicode_escape')
    mr4_df = pd.read_csv(f"../{bot_name}/24-MR4/evaluated answers/tested_MR4_pair.csv", encoding='unicode_escape')
    mr5_1_df = pd.read_csv(f"../{bot_name}/26-MR5_someall/evaluated answers/tested_MR5_someall_pair.csv", encoding='unicode_escape')
    mr5_2_df = pd.read_csv(f"../{bot_name}/25-MR5_allsome/evaluated answers/tested_MR5_allsome_pair.csv", encoding='unicode_escape')

    methods={"basic": find_category_bias_in_df(basic_df, bot_name, single=False),
             "MR1": find_category_bias_in_df(mr1_df, bot_name, single=False),
             "MR2": find_category_bias_in_df(mr2_df, bot_name, single=False),
             "mr1_mr3_df": find_category_bias_in_df(mr1_mr3_df, bot_name, single=False),
             "mr1_mr4_df": find_category_bias_in_df(mr1_mr4_df, bot_name, single=False),
             "mr1_mr5_1_df": find_category_bias_in_df(mr1_mr5_1_df, bot_name, single=False),
             "mr1_mr5_2_df": find_category_bias_in_df(mr1_mr5_2_df, bot_name, single=False),
             "mr2_mr3_df": find_category_bias_in_df(mr2_mr3_df, bot_name, single=False),
             "mr2_mr4_df": find_category_bias_in_df(mr2_mr4_df, bot_name, single=False),
             "mr2_mr5_1_df": find_category_bias_in_df(mr2_mr5_1_df, bot_name, single=False),
             "mr2_mr5_2_df": find_category_bias_in_df(mr2_mr5_2_df, bot_name, single=False),
             "mr1_mr2_df": find_category_bias_in_df(mr1_mr2_df, bot_name, single=False),
             "mr1_mr2_mr3_df": find_category_bias_in_df(mr1_mr2_mr3_df, bot_name, single=False),
             "mr1_mr2_mr4_df": find_category_bias_in_df(mr1_mr2_mr4_df, bot_name, single=False),
             "mr1_mr2_mr5_1_df": find_category_bias_in_df(mr1_mr2_mr5_1_df, bot_name, single=False),
             "mr1_mr2_mr5_2_df": find_category_bias_in_df(mr1_mr2_mr5_2_df, bot_name, single=False),
             "mr3_df": find_category_bias_in_df(mr3_df, bot_name, single=False),
             "mr4_df": find_category_bias_in_df(mr4_df, bot_name, single=False),
             "mr5_1_df": find_category_bias_in_df(mr5_1_df, bot_name, single=False),
             "mr5_2_df": find_category_bias_in_df(mr5_2_df, bot_name, single=False)}

    # for key in methods.keys():
    #     print(key, methods[key])
    return methods

def find_category_bias_single(bot_name):
    basic_df = pd.read_csv(f"../{bot_name}/1-basic/evaluated answers/tested_basic_single.csv", encoding='unicode_escape')
    mr1_df = pd.read_csv(f"../{bot_name}/2-MR1/evaluated answers/tested_MR1_single.csv", encoding='unicode_escape')
    mr2_df = pd.read_csv(f"../{bot_name}/3-MR2/evaluated answers/tested_MR2_single.csv", encoding='unicode_escape')
    mr1_mr3_df = pd.read_csv(f"../{bot_name}/10-MR1_somesome/evaluated answers/tested_MR1_some_single.csv", encoding='unicode_escape')
    mr1_mr4_df = pd.read_csv(f"../{bot_name}/11-MR1_allall/evaluated answers/tested_MR1_all_all_single.csv", encoding='unicode_escape')
    mr2_mr3_df = pd.read_csv(f"../{bot_name}/14-MR2_somesome/evaluated answers/tested_MR2_some_single.csv", encoding='unicode_escape')
    mr2_mr4_df = pd.read_csv(f"../{bot_name}/15-MR2_all_all/evaluated answers/tested_MR2_all_single.csv", encoding='unicode_escape')
    mr1_mr2_df = pd.read_csv(f"../{bot_name}/18-MR1_MR2/evaluated answers/tested_MR1_MR2_single.csv", encoding='unicode_escape')
    mr1_mr2_mr3_df = pd.read_csv(f"../{bot_name}/19-MR1_MR2_somesome/evaluated answers/tested_MR1_MR2_some_single.csv", encoding='unicode_escape')
    mr1_mr2_mr4_df = pd.read_csv(f"../{bot_name}/20-MR1_MR2_allall/evaluated answers/tested_MR1_MR2_all_single.csv", encoding='unicode_escape')
    mr3_df = pd.read_csv(f"../{bot_name}/23-MR3/evaluated answers/tested_MR3_single.csv", encoding='unicode_escape')
    mr4_df = pd.read_csv(f"../{bot_name}/24-MR4/evaluated answers/tested_MR4_single.csv", encoding='unicode_escape')

    methods={"basic": find_category_bias_in_df(basic_df, bot_name),
             "MR1": find_category_bias_in_df(mr1_df, bot_name),
             "MR2": find_category_bias_in_df(mr2_df, bot_name),
             "mr1_mr3_df": find_category_bias_in_df(mr1_mr3_df, bot_name),
             "mr1_mr4_df": find_category_bias_in_df(mr1_mr4_df, bot_name),
             "mr2_mr3_df": find_category_bias_in_df(mr2_mr3_df, bot_name),
             "mr2_mr4_df": find_category_bias_in_df(mr2_mr4_df, bot_name),
             "mr1_mr2_df": find_category_bias_in_df(mr1_mr2_df, bot_name),
             "mr1_mr2_mr3_df": find_category_bias_in_df(mr1_mr2_mr3_df, bot_name),
             "mr1_mr2_mr4_df": find_category_bias_in_df(mr1_mr2_mr4_df, bot_name),
             "mr3_df": find_category_bias_in_df(mr3_df, bot_name),
             "mr4_df": find_category_bias_in_df(mr4_df, bot_name)}    

    return methods
def concat_dicts(dict1, dict2, dict3):
    final_dict=dict1.copy()
    
    for method in final_dict.keys():
        categories = final_dict[method]
        for category in categories.keys():
            final_dict[method][category] += dict2[method][category] + dict3[method][category]
    
    return final_dict
    
df_llama2 = dataframe[dataframe['Bot']=='llama2']
df_dialogpt = dataframe[dataframe['Bot']=='dialogpt']
df_gpt = dataframe[dataframe['Bot']=='Gpt 3.5 turbo']


# show_simple_comparison_results(bot_name="dialogpt", df=df_dialogpt, question_type="pair")
# show_simple_comparison_results(bot_name="dialogpt", df=df_dialogpt, question_type="single")
# show_simple_comparison_results(bot_name="llama2", df=df_llama2, question_type="pair")
# show_simple_comparison_results(bot_name="llama2", df=df_llama2, question_type="single")
# show_simple_comparison_results(bot_name="gpt 3.5 turbo", df=df_gpt, question_type="pair")
# show_simple_comparison_results(bot_name="llama2", df=df_gpt, question_type="single")

# show_explicit_implicit_questions_simple_comparision_results(dataframe, "Pair")
# show_explicit_implicit_questions_simple_comparision_results(dataframe, "Single")

# df_dialogpt_basic_pair = pd.read_csv("../gpt-3.5-turbo/1-basic/evaluated answers/tested_basic_single.csv", encoding='unicode_escape')
# df_dialogpt_additional_pair = pd.read_csv("../gpt-3.5-turbo/24-MR4/evaluated answers/tested_MR4_single_old.csv", encoding='unicode_escape')
# calculate_new_bias(df_dialogpt_basic_pair, df_dialogpt_additional_pair)

# draw_bias_ditribution_plot_pair("dialogpt")
# draw_bias_ditribution_plot_single("dialogpt")
# draw_bias_ditribution_plot_pair("llama2")
# draw_bias_ditribution_plot_single("llama2")
# draw_bias_ditribution_plot_pair("gpt-3.5-turbo")
# draw_bias_ditribution_plot_single("gpt-3.5-turbo")


# show_shared_comparison_results(df=df_dialogpt, question_type="pair", sort_plots=True, show_combinations=True)
# show_shared_comparison_results(df=df_dialogpt, question_type="single", sort_plots=True, show_combinations=True)
# show_shared_comparison_results(df=df_llama2, question_type="pair", sort_plots=True, show_combinations=True)
# show_shared_comparison_results(df=df_llama2, question_type="single", sort_plots=True, show_combinations=True)
# show_shared_comparison_results(df=df_gpt, question_type="pair", sort_plots=True, show_combinations=True)
# show_shared_comparison_results(df=df_gpt, question_type="single", sort_plots=True, show_combinations=True)

# show_explicit_implicit_questions_shared_comparision_results(dataframe, question_type="Pair")
# show_explicit_implicit_questions_shared_comparision_results(dataframe, question_type="Single")

draw_stock_plot_for_adding_all_MRs()

# dialogpt_cat_pair = find_category_bias_pair("dialogpt")
# llama2_cat_pair = find_category_bias_pair("llama2")
# gpt_cat_pair = find_category_bias_pair("gpt-3.5-turbo")
# categories_pair = concat_dicts(dialogpt_cat_pair, llama2_cat_pair, gpt_cat_pair)
# for method in categories_pair.keys():
#     print(method)
#     print(categories_pair[method])
#     print("-----------")
# find_category_bias_single("dialogpt")