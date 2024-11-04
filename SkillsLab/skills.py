import randomx
from variable import Variable
from probFactors import Prob
from probGraphicalModels import BeliefNetwork
import probRC
import probStochSim

random.seed(23)

# Corresponding with the examples in modules 9-10 (specifically 9.3 and 10.1), we will create a belief
# network of students and their grades (see Figure 10A on ZyBooks page 10.1).
# 
# The first step is to create the variables of interest.  Recall that each variable has a given
# domain - the set of values the variable can have.  In this case, we will just use "true" and "false".
# In Python, we will implement this as a set.  Note we can have other sets of domain variables, but for this
# example we will just use the following - uncomment the line to enable it.

boolean = [False, True]

# Next, we will setup the variables themselves.  Example 9J in Zybooks (page 9.3 of Module 9) provides 
# four variables.  We can define these variables with the constructor "Variable" in AIPython that has two
# arguments - the name and the domain.  An additional position argument is optional.  We complete one of the 
# variables below and provide the name of the others.

Intelligent = Variable("Intelligent", boolean)
Works_hard = Variable("Works_hard", boolean)
Answers = Variable("Answers", boolean)
Grade = Variable("Grade", boolean)

# Next we need to setup the priors, conditional probabilities and depdendency relationships among variables.
# Example 10A has the depdendency relationships.  We will go through this step-by-step.
# First, we will setup the prior probabilitieis.  This is the same as setting up conditionals with no
# parents.  We will use the following priors:
#      P(Intelligent=True) = 0.85
#      P(Works_hard=True) = 0.67
# We write the code for the first below.

prior_Intelligent = Prob(Intelligent,[],[0.15,0.85])

# Note the empty list (this rerpresnets that the node has no parents).
# The other list represents a probability distirbution over the domain (in this case, "boolean" - which we
# defined earlier as a list and associated with teh variable earlier).  Uncomment and finish the code for "Works_hard" below.

prior_Works_hard =  Prob(Works_hard, [], [0.33, 0.67])

# Now we need to setup the conditionals.  We will use the following, which follows the structure of Example 10A.
# (Note Example 10A does not give prior or conditionals):
#     P(Answers=True | Works_hard=False, Intelligent = False) = 0.27
#     P(Answers=True | Works_hard=False, Intelligent = True) = 0.65
#     P(Answers=True | Works_hard=True, Intelligent=False) = 0.4
#     P(Answers=True | Works_hard=True, Intelligent=True) = 0.72
#     P(Grade=True | Answers=True) = 1.0
#     P(Grade=True | Answers=False) = 0.0
# We will write out the first two below in code:

cond_Answers = Prob(Answers,[Works_hard, Intelligent],[[[0.73,0.27],
                                                        [0.35,0.65]],
                                                       [[0.6,0.4],
                                                        [0.28,0.72]]])

# The below comment is incorrect Python syntax - but may help some students understand: we replace the values with the conditional probabilities.

# cond_Answers = Prob(Answers,[Works_hard, Intelligent],[[[P(Answers=False | Works_hard=False, Intelligent = False),P(Answers=True | Works_hard=False, Intelligent = False)],
#                                                         [P(Answers=False | Works_hard=False, Intelligent = True),P(Answers=True | Works_hard=False, Intelligent = True)],
#                                                        [[P(Answers=False | Works_hard=True, Intelligent = False),P(Answers=True | Works_hard=True, Intelligent = False)],
#                                                         [P(Answers=False | Works_hard=True, Intelligent = True),P(Answers=True | Works_hard=True, Intelligent = True)]]])


# Note the above line of code specifies all conditionals relating to Answers based on its parents.
# In the first list (the second argument) we list all the parents.  The second list (is a list of lists)
# and it specifies the probaiblity distribution of "Answers" given certain conditions.  Note there is one
# list for each variable that "Answers" is conditioned on.  For example, [0.65,0.35] is the probability 
# distribution for Answers (true or false) given that "Works_hard" is true.  Likewise, [0.27,0.73] is 
# the distribution for "Asnwers" given that "Works_hard" is false. 
# Below, please complete the code for the conditional for "Grade":

cond_Grade = Prob(Grade, [Answers], [
    [1.0, 0.0],   
    [0.0, 1.0]    ]) 
#cond_Grade = Prob(Grade,[Answers], [[P(Grade=False | Answers = False),P(Grade=True | Answers = False)],
#                                    [P(Grade=False | Answers = True), P(Grade=True | Answers = True)]])



# Now that we have all the components, we can establish the belief network.  Uncomment the below line
# to enable the code.

student_bn = BeliefNetwork("Student Grades", {Works_hard, Intelligent, Answers, Grade}, {prior_Works_hard, prior_Intelligent, cond_Answers, cond_Grade})

# You can display the belief network graphcially with the .show method.  Uncomment the below line to try it out.

# student_bn.show()

# Now that we have a belief network, we can answer queries not directly represneted in the model.  In this ICE,
# we will computer P(Grade = True | Intelligent=True & Works_hard=True).  We will look at three methods.

# The first method is an exact computation using search.  This is OK to use for this small problem, but note it has exponential runtime
# so it is not suitable for actual problems.  We use the AI Python impplementation below creating an object with the "ProbSearch" constructor.
# Uncomment the folloiwng line to activate it.

infer = probRC.ProbSearch(student_bn)

# Now we want to do a query using that algorithm on our belief network using the query method.  Notice how the first argument is the 
# variable of interest, and the other arument is a dictionary that has the assignment of all variables it is conditioned on.
# Uncomment the following line to activate and print to the screen.

q1 = infer.query(Grade,{Intelligent: True, Works_hard: False})
print(q1)

# This query should return the same probability of P(Answers=True | Works_hard=False, Intelligent = True) in the spec - that's because we
# have a query that is the same as the spec, so we would get this result.  However, if we change the evidence, the results change.

q2 = infer.query(Grade,{Intelligent: True})
print(q2)

# Note the value changes, as now it is considering the prior P(Works_hard = True)

# The next algorithm is recursive conditioning.  The constructor has the same syntax, but is called "ProbRC" (also in probRC.py).
# Uncomment and complete the below line of code.

inferRC = probRC.ProbRC(student_bn)


# We can also use the query method here - it works the same as for ProbSearch.  
# Uncomment and complete the below line of code for the query P(Grade = True | Intelligent = True).

q3 = inferRC.query(Grade, {Intelligent: True})
print(q3)


# As described in Module 12, stochastic simulation offers an alternative to exact methods.  In this ICE, we provide the syntax for two of 
# the methods from the course.  First we have rejection sampling.  Note we can adjust the number of samples.  In the folloiwng, implement this with 10,
# 100, 1000, and 10000 samples and note the difference in the output.

rc = probStochSim.RejectionSampling(student_bn)
q4 = rc.query(Grade,{Intelligent: True}, number_samples= 10)
q5 = rc.query(Grade,{Intelligent: True}, number_samples= 100)
q6 = rc.query(Grade,{Intelligent: True}, number_samples= 1000)
q7 = rc.query(Grade,{Intelligent: True}, number_samples= 10000)

print(q4)
print(q5)
print(q6)
print(q7)

# Note how the probaiblity gets closer to the exact solution as you add samples.

# We can also do this for liklihood sampling, which we show in our final example below.

lw = probStochSim.LikelihoodWeighting(student_bn)
q8=lw.query(Grade,{Intelligent: True}, number_samples = 100)
print(q8)
