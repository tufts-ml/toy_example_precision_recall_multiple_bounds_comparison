

## Optimizing Early Warning Classifiers to Control False Alarms by Satisfying a Minimum Precision Constraint

This repo is for an anonymous AISTATS submission, designed to help the reader to investigate the toy examples in the paper. 

We provide a notebook comparing 4 different methodologies that are trained to maximize recall, subject to precision>= 0.9. 
* BCE + threshold search
* Eban et al's hinge bound
* Fathony & Kolter's adversarial prediction bound
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

   - **BCE + Threshold search :** We first try Binary cross entropy + post-hoc threshold search, which is commonly used in many applications for the meeting the desired precision-recall. 

![](images/BCE_plus_threshold_search_solution.png?raw=true)
    
   However, we see that even with post-hoc search, BCE cannot achieve the desired precision.

   - **Eban et al's hinge bound :** We then try [Eban et al's](http://proceedings.mlr.press/v54/eban17a/eban17a.pdf) proposed hinge bound.  
   
   
![](images/hinge_solution_precision_90.png?raw=true)


   Here again, the hinge bound falls short of the desired precision of 0.9, reaching 0.79 instead. We hypothesize that this is due to the looseness of the hinge bound.

   - **Fathony & Kolter's adversarial prediction bound :** Next, we try the optimizing our custom objective using an adversarial prediction framework recent proposed by [Fathony & Kolter](http://proceedings.mlr.press/v108/fathony20a.html).

![](images/adversarial_prediction_precision_90.png?raw=true)

   The adversarial prediction bound reached the desired precision of 0.9, and is able to achieve a recall of 0.11, without any post-hoc threshold search. However the total runtime is nearly 3000 seconds, which is 300x the training time required for the other 3 methods.

   - **Our proposed sigmoid bound :** Finally, we show the decision boundary of our proposed sigmoid bound, which is tight, differentiable, making gradient-based learning feasible.
   
![](images/sigmoid_solution_precision_90.png?raw=true)

   Our proposed sigmoid bound reaches the desired precision of 0.9, without any post-hoc threshold search, and achieves a recall of 0.23, which is nearly 2x the recall achieved by Fathony & Kolter's adversarial prediction bound. Moreover our proposed bound requires a training time of ~15 seconds, which is $(1/300)^{th}$ of the training time required by Fathony & Kolter's adversarial prediction bound.

