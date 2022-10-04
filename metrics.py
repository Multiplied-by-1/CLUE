import sys
def calculate_acc(results, file):
    task0_acc = results[3]['Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000']
    task1_acc = results[3]['Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp001']
    task2_acc = results[3]['Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp002']
    task3_acc = results[3]['Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp003']
    task_mean_acc = (task0_acc + task1_acc + task2_acc + task3_acc) / 4
    file_path = file
    sys.stdout = open(file_path, "a")
    print('=' * 80)
    print('Task mean accuracy:', task_mean_acc)
    return task_mean_acc

def calculate_forget(results, file):
    task0_top = results[0]['Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000']
    task1_top = results[1]['Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp001']
    task2_top = results[2]['Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp002']
    task0_last = results[3]['Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000']
    task1_last = results[3]['Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp001']
    task2_last = results[3]['Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp002']
    task_mean_forget = ((max(0, task0_top - task0_last) / task0_top) + (max(0, task1_top - task1_last) / task1_top) + \
                       (max(0, task2_top - task2_last) / task2_top)) / 3
    file_path = file
    sys.stdout = open(file_path, "a")
    print('=' * 80)
    print('Task mean forgetting:', task_mean_forget)
    return task_mean_forget