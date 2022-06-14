# Python program to analyze the .json files produced by running main.py
# pythonw main.py val 2 m 11 greedy true nuscenes results/000008
import json
import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt

def plot_results(per_sample_results, agg_dict):
  # Total number of unique objects
  num_obj = len(agg_dict) - 1
  # Plot per object
  #gs = gridspec.GridSpec(num_obj,2)
  gs = gridspec.GridSpec(2, 1)
  axs = [plt.subplot(g) for g in gs]
  # Number of samples
  num_samples = len(per_sample_results)
  t = np.arange(num_samples)

  # Create the object information per sample
  obj_info = np.zeros((num_obj, num_samples, 2))
  for t_ind in range(num_samples):
    # This is all the unique number of objects in the aggregated dictionary
    for obj_ind, (obj_name, obj_value) in enumerate(agg_dict.items()):
        # Exclude the total number of items, collect the information for all the other objects
        if (obj_name != "total items per sample"):
            obj_info[obj_ind-1,t_ind,:] = per_sample_results[t_ind].get(obj_name, np.array([0.0, 0.0]))

  # plot per object
  plot_ind = 0
  for obj_ind, (obj_name, obj_value) in enumerate(agg_dict.items()):
    if obj_name != "avg items per sample":
        if obj_name == "bicycle":
            axs[plot_ind].plot(t, obj_info[obj_ind-1,:,0], 'b', label='Items')
            axs[plot_ind].set_ylabel('Num')
            axs[plot_ind].set_xlabel('Samples')
            title_str = "Number of " + obj_name + "s per sample"
            axs[plot_ind].set_title(title_str)
            #axs[plot_ind].set_aspect('equal')

            plot_ind += 1

            axs[plot_ind].plot(t, obj_info[obj_ind-1,:,1], 'g', label='Scores')
            axs[plot_ind].set_ylabel('Score')
            axs[plot_ind].set_xlabel('Samples')
            title_str = obj_name + " AMOTA per sample"
            axs[plot_ind].set_title(title_str)
            #axs[plot_ind].set_aspect('equal')

        plot_ind +=1
    print(obj_ind, plot_ind, obj_name)
  plt.show()

# Results file
data_dir = '000008'
results_dir = '/Users/mho/Documents/StanfordClasses/AA273/Project/stereo_based_tracking/results/' + data_dir + '/val/'
file_name = 'results_val_probabilistic_tracking.json'
path_to_file = results_dir + file_name
with open(path_to_file, 'r') as f:
  data_dict = json.load(f)

# Get out the results
# Form a list of dictionaries to collect results
per_sample_results = []
obj = data_dict["results"]
agg_dict = {"avg items per sample": np.array([0.0, 0.0])}
# Per sample
for i, (k, v) in enumerate(obj.items()):
# i is the index, k is the sample_token corresponding to the jpg frame, v is the value corresponding to a list of (e.g., 90) objects
# Each v object is a dictionary of length 8 (sample_token, translation, size, rotation, velocity, tracking_id, tracking_name, tracking_score
# Get the samples, and give values as (number of objects, aggregate tracking scores)
    sample_dict = {"total items per sample": np.array([0, 0.0])}
    # Go through every object in the sample
    for obj_ind in range(len(v)):
      # Get the object name
      obj_name = v[obj_ind].get('tracking_name')
      obj_score = v[obj_ind].get('tracking_score')
      # If not in the dictionary, it will be added and items incremented
      sample_dict[obj_name] = sample_dict.get(obj_name, np.array([0,0.0])) + np.array([1,obj_score])
      # If not in the dictionary, it will be added and items incremented
      agg_dict[obj_name] = agg_dict.get(obj_name, np.array([0, 0.0])) + np.array([1, obj_score])

#    print("loop 1, hi", i)
    # When all done with gathering objects, go through and average across a sample:
    for obj_ind, (obj_name, obj_value) in enumerate(sample_dict.items()):
      # Add to the total number of items and total scores per sample (don't bother to average)
      sample_dict["total items per sample"] = sample_dict.get("total items per sample") + sample_dict[obj_name]
      # Compute the average of all the scores per object, since these are unique - including the total
      if (sample_dict[obj_name][0] > 0 and obj_name != "total items per sample"):
        sample_dict[obj_name][1] = sample_dict.get(obj_name)[1]/sample_dict.get(obj_name)[0]

    # Final average before appending
    sample_dict["total items per sample"][1] = sample_dict.get("total items per sample")[1] / sample_dict.get("total items per sample")[0]

    # Total items
    agg_dict["avg items per sample"] = agg_dict.get("avg items per sample") + sample_dict.get("total items per sample")

#    print("loop 2, hi", obj_ind)
    # When done with all the objects in the sample_token, then append the dictionary
    per_sample_results.append(sample_dict)
#   print("hi")

for obj_ind, (obj_name, obj_value) in enumerate(agg_dict.items()):
  agg_dict[obj_name] = agg_dict.get(obj_name)/float(len(per_sample_results))

print(agg_dict)
for obj_ind, (obj_name, obj_value) in enumerate(agg_dict.items()):
    print(obj_ind, obj_name, obj_value)

plot_results(per_sample_results, agg_dict)
# Print the results
print("hi")