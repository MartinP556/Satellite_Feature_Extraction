import numpy as np

def phase_dependent_response(driver_values, t_dev, responses, thresholds):
    #Thresholds are the thresholds in development time where the different growth phases change
    #Responses are the response functions, index starting at 'before the first threshold'
    #driver values are the inputs to the response function
    #t_dev is the (cts) development time
    phase = np.digitize(t_dev, thresholds)
    response = np.zeros(driver_values.shape)
    for phase_index in range(len(responses)):
        response += (phase == phase_index)*responses[phase_index](driver_values) #First brackets indicates if we are at the right phase, second takes the response function for each phase
    return response

def Wang_Engel_Temp_response(T, T_min, T_opt, T_max, beta = 1):
    alpha = np.log(2)/np.log( (T_max - T_min)/(T_opt - T_min) )
    f_T = ( ( (2*(T - T_min)**alpha)*((T_opt - T_min)**alpha) - ((T - T_min)**(2*alpha)) ) / ((T_opt - T_min)**(2*alpha)) )**beta
    f_T = np.nan_to_num(f_T)
    return f_T*(T >= T_min)*(T<= T_max)

def Trapezoid_Temp_response(T, T_min, T_opt1, T_opt2, T_max):
    pre_opt = (T>=T_min)*(T<=T_opt1)
    opt = (T>=T_opt1)*(T<=T_opt2)
    post_opt = (T>=T_opt2)*(T<=T_max)
    return pre_opt*(T - T_min)/(T_opt1 - T_min) + opt + post_opt*(T_max - T)/(T_max - T_opt2)

def double_logistic(x, a, b, c, d, e, f):
    return a / (1 + np.exp(-b * (x - c))) + d / (1 + np.exp(-e * (x - f)))

def normalized_difference(x1, x2):
    return (x1 - x2) / (x1 + x2)