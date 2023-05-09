from sympy import parse_expr, solve, pi, sin, cos, tan


def get_num(user_input):
    try:
        expr = parse_expr(user_input)
        result = expr.evalf(
            subs={'pi': pi, 'sin': sin, 'cos': cos, 'tan': tan})
        return result
    except:
        try:
            x = user_input.split()
            if len(x) == 3:
                if x[1] == "%":
                    result = int(x[0]) % int(x[2])
                elif x[1] == "**":
                    result = int(x[0]) ** int(x[2])
                elif x[1] == "root":
                    result = int(x[2]) ** (1/int(x[0]))
                else:
                    angle = float(x[0]) * (pi / 180)
                    if x[1] == "sin":
                        result = sin(angle)
                    elif x[1] == "cos":
                        result = cos(angle)
                    elif x[1] == "tan":
                        result = tan(angle)
                return result
            elif "solve" in user_input:
                eq = user_input.replace("solve", "")
                eq = parse_expr(eq)
                sol = solve(eq)
                if len(sol) == 0:
                    return "Sorry, no solutions found."
                elif len(sol) == 1:
                    return "The solution is:", sol[0]
                else:
                    return "There are multiple solutions:"
            else:
                return "Sorry, I couldn't understand that. Please try again."
        except:
            return "Sorry, I couldn't understand that. Please try again."
