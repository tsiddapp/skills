import randomx
from variable import Variable
from probFactors import Prob
from probGraphicalModels import BeliefNetwork
import probRC
import probStochSim

random.seed(23)

boolean = [False, True]

Intelligent = Variable("Intelligent", boolean)
Works_hard = Variable("Works_hard", boolean)
Answers = Variable("Answers", boolean)
Grade = Variable("Grade", boolean)

prior_Intelligent = Prob(Intelligent,[],[0.15,0.85])

prior_Works_hard =  Prob(Works_hard, [], [0.33, 0.67])

cond_Answers = Prob(Answers,[Works_hard, Intelligent],[[[0.73,0.27],
                                                        [0.35,0.65]],
                                                       [[0.6,0.4],
                                                        [0.28,0.72]]])
cond_Grade = Prob(Grade, [Answers], [
    [1.0, 0.0],   
    [0.0, 1.0]    ]) 

student_bn = BeliefNetwork("Student Grades", {Works_hard, Intelligent, Answers, Grade}, {prior_Works_hard, prior_Intelligent, cond_Answers, cond_Grade})

infer = probRC.ProbSearch(student_bn)

q1 = infer.query(Grade,{Intelligent: True, Works_hard: False})
print(q1)

q2 = infer.query(Grade,{Intelligent: True})
print(q2)

inferRC = probRC.ProbRC(student_bn)

q3 = inferRC.query(Grade, {Intelligent: True})
print(q3)

rc = probStochSim.RejectionSampling(student_bn)
q4 = rc.query(Grade,{Intelligent: True}, number_samples= 10)
q5 = rc.query(Grade,{Intelligent: True}, number_samples= 100)
q6 = rc.query(Grade,{Intelligent: True}, number_samples= 1000)
q7 = rc.query(Grade,{Intelligent: True}, number_samples= 10000)

print(q4)
print(q5)
print(q6)
print(q7)
lw = probStochSim.LikelihoodWeighting(student_bn)
q8=lw.query(Grade,{Intelligent: True}, number_samples = 100)
print(q8)
