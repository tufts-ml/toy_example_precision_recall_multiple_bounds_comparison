
## Optimizing Early Warning Classifiers to Control False Alarms by Satisfying a Minimum Precision Constraint

This repo is for an anonymous AISTATS submission, designed to help the reader to investigate the toy examples in the paper. 

We provide a notebook comparing 4 different methodologies that are trained to maximize recall, subject to precision>= 0.9. 
* BCE + threshold search
* Eban et al's hinge bound
* Fathony et al's adversarial prediction bound
* Our proposed sigmoid bound

### Workflow

 1. Users are expected to installing the conda enviroment via the [toy_false_alarm_control.yml](toy_false_alarm_control.yml) file provided

```python
>> conda env create --name toy_false_alarm_control --file=toy_false_alarm_control.yml
>> conda activate toy_false_alarm_control
```

2. Open the [notebook for reproducing and comparing multiple bounds on the toy example](toy_example_comparing_BCE_Hinge_and_Sigmoid.ipynb) 

3. Run the cells to create the toy example. Our toy example is heavily imbalanced with 120 positive examples and 450 negative examples.
![](images/toy_example.png?raw=true)

 4. Our goal is to train a linear classifier to find a decision boundary that maximizes recall subject to precision>=0.9.

	 - We first try Binary cross entropy + post-hoc threshold search, which is commonly used in many applications for the meeting the desired precision-recall. However, we see that even with post-hoc search, BCE cannot achieve the desired precision
	 ![](images/BCE_plus_threshold_search_solution.png?raw=true)
 
