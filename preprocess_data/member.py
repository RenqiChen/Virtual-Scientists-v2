import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# read txt as array
team_list = np.loadtxt('/home/bingxing2/ailab/scxlab0066/SocialScience/team_list_new.txt')

# draw the distribution
print(np.max(team_list))
print(np.min(team_list))
team_list = team_list - 3
team_list = team_list[team_list <= 40]

# normalize the histogram to get a probability distribution
plt.figure()
counts, bins, patches = plt.hist(team_list, bins=20, color='blue', alpha=0.7, density=True)
plt.xlabel('Number of authors')
plt.ylabel('Probability')
plt.title('Distribution of number of authors in papers')
plt.show()

# save the figure as png format
plt.savefig('/home/bingxing2/ailab/scxlab0066/SocialScience/Social_Science_CAMEL/preprocess_data/team_member.png')

# use an exp function to fit the data
def func(x, a, b, c):
    return a * np.exp(-b * x) + c

# fit the data
x = np.arange(0, 51)
y, _ = np.histogram(team_list, bins=51, density=True)
popt, pcov = curve_fit(func, x[:len(y)], y, maxfev=20000)

# draw the fitting curve
plt.figure(figsize=(10, 6)) 
plt.plot(x, func(x, *popt), 'r-', label='Fit: %5.3f * np.exp(-%5.3f * (x-3)) + %5.3f' % tuple(popt))
plt.hist(team_list, bins=40, color='blue', alpha=0.7, density=True)
plt.xticks(np.arange(0, 44, step=5), labels=np.arange(3, 47, step=5))
plt.xlabel('Team Size',fontsize = 20)
plt.ylabel('Probability', fontsize=20)
plt.xlim(0,40)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.show()

# print a, b, c, respectively
print(popt[0], popt[1], popt[2])

# save the figure as png format
plt.savefig('/home/bingxing2/ailab/scxlab0066/SocialScience/Social_Science_CAMEL/preprocess_data/team_member_fit.png')
# save the figure as high-resolution PDF
plt.savefig('/home/bingxing2/ailab/scxlab0066/SocialScience/Social_Science_CAMEL/preprocess_data/team_member_fit.pdf', format='pdf', dpi=300)