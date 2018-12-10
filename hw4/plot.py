# import modules
import os
import pandas
import argparse
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--exps',  nargs='+', type=str)
parser.add_argument('--save', type=str, default=None)
args = parser.parse_args()

# initialize figure
f, ax = plt.subplots(1, 1)

# generate plots for all experiments
for i, exp in enumerate(args.exps):
    log_fname = os.path.join('data', exp, 'log.csv')
    csv = pandas.read_csv(log_fname)

    color = cm.viridis(i / float(len(args.exps)))
    ax.plot(csv['Itr'], csv['ReturnAvg'], color=color, label=exp)
    ax.fill_between(csv['Itr'], csv['ReturnAvg'] - csv['ReturnStd'], csv['ReturnAvg'] + csv['ReturnStd'],
                    color=color, alpha=0.2)

# add xlabel, ylabel and legend
ax.legend()
ax.set_ylabel('Return')
ax.set_xlabel('Iteration')

# save plots or visualize
if args.save:
    os.makedirs('plots', exist_ok=True)
    f.savefig(os.path.join('plots', args.save + '.jpg'))
else:
    plt.show()
