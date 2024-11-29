import numpy as np
# read txt as array
team_list = np.loadtxt('/home/bingxing2/ailab/scxlab0066/SocialScience/team_list.txt')
# draw the distribution
import matplotlib.pyplot as plt
print(np.max(team_list))
team_list = team_list-3
# display the range between 1 and 10
team_list = team_list[team_list<=40]
plt.figure()
plt.hist(team_list, bins=20, color='blue', alpha=0.7)
plt.xlabel('Number of authors')
plt.ylabel('Number of papers')
plt.title('Distribution of number of authors in papers')
plt.show()
# save the figure as png format
plt.savefig('/home/bingxing2/ailab/scxlab0066/SocialScience/Social_Science_CAMEL/preprocess_data/team_member.png')
# use a exp function to fit the data
from scipy.optimize import curve_fit
def func(x, a, b, c):
    return a * np.exp(-b * x) + c
# fit the data
x = np.arange(1, 51)
y = np.histogram(team_list, bins=50)[0]
popt, pcov = curve_fit(func, x, y, maxfev=10000)
# draw the fitting curve
plt.figure()
plt.plot(x, func(x, *popt), 'r-', label='fit: %5.3f * np.exp(-%5.3f * x) + %5.3f' % tuple(popt))
plt.hist(team_list, bins=40, color='blue', alpha=0.7)
plt.xlabel('Number of authors')
plt.ylabel('Number of papers')
plt.title('Distribution of number of authors in papers')
plt.legend()
plt.show()
# print a,b,c, respectively
print(popt[0], popt[1], popt[2])
# save the figure as png format
plt.savefig('/home/bingxing2/ailab/scxlab0066/SocialScience/Social_Science_CAMEL/preprocess_data/team_member_fit.png')
