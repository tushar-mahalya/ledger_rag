# ruff: noqa
# mypy: ignore-errors
"""Official FinQA execution semantics, vendored verbatim (MIT, see LICENSE.finqa).

Source: https://github.com/czyssrs/FinQA/blob/0f16e2867befa6840783e58be38c9efb9229d742/code/evaluate/evaluate.py
File SHA-256 845cd131cab843eceff256cf6d392978cc470a7da4a80107beb56027fdca5c13.
Copied unmodified: ``all_ops`` (lines 24-25), ``str_to_num``, ``process_row`` and
``eval_program`` (lines 27-194) and ``program_tokenization`` (lines 368-386). The
symbolic ``equal_program`` (program accuracy) is omitted because it needs SymPy.
Do not edit: execution accuracy is only comparable to published FinQA numbers while
this code matches upstream. ``tests/unit/test_official_finqa.py`` pins the hash.
"""

all_ops = ["add", "subtract", "multiply", "divide", "exp", "greater", "table_max", \
"table_min", "table_sum", "table_average"]

def str_to_num(text):
    
    text = text.replace(",", "")
    try:
        num = float(text)
    except ValueError:
        if "%" in text:
            text = text.replace("%", "")
            try:
                num = float(text)
                num = num / 100.0
            except ValueError:
                num = "n/a"
        elif "const" in text:
            text = text.replace("const_", "")
            if text == "m1":
                text = "-1"
            num = float(text)
        else:
            num = "n/a"
    return num

def process_row(row_in):
    
    row_out = []
    invalid_flag = 0
    
    for num in row_in:
        num = num.replace("$", "").strip()
        num = num.split("(")[0].strip()
        
        num = str_to_num(num)
        
        if num == "n/a":
            invalid_flag = 1
            break
        
        row_out.append(num)
        
    if invalid_flag:
        return "n/a"
    
    return row_out


def eval_program(program, table):
    '''
    calculate the numerical results of the program
    '''

    invalid_flag = 0
    this_res = "n/a"
    
    try:
        program = program[:-1] # remove EOF
        # check structure
        for ind, token in enumerate(program):
            if ind % 4 == 0:
                if token.strip("(") not in all_ops:
                    return 1, "n/a"
            if (ind + 1) % 4 == 0:
                if token != ")":
                    return 1, "n/a"


        program = "|".join(program)
        steps = program.split(")")[:-1]
        
        
        res_dict = {}
        
        # print(program)
        
        for ind, step in enumerate(steps):
            step = step.strip()
            
            if len(step.split("(")) > 2:
                invalid_flag = 1
                break
            op = step.split("(")[0].strip("|").strip()
            args = step.split("(")[1].strip("|").strip()
            
            # print(args)
            # print(op)
            
            arg1 = args.split("|")[0].strip()
            arg2 = args.split("|")[1].strip()
            
            if op == "add" or op == "subtract" or op == "multiply" or op == "divide" or op == "exp" or op == "greater":
                
                if "#" in arg1:
                    arg1 = res_dict[int(arg1.replace("#", ""))]
                else:
                    # print(arg1)
                    arg1 = str_to_num(arg1)
                    if arg1 == "n/a":
                        invalid_flag = 1
                        break
                
                if "#" in arg2:
                    arg2 = res_dict[int(arg2.replace("#", ""))]
                else:
                    arg2 = str_to_num(arg2)
                    if arg2 == "n/a":
                        invalid_flag = 1
                        break
                
                if op == "add":
                    this_res = arg1 + arg2
                elif op == "subtract":
                    this_res = arg1 - arg2
                elif op == "multiply":
                    this_res = arg1 * arg2
                elif op == "divide":
                    this_res = arg1 / arg2
                elif op == "exp":
                    this_res = arg1 ** arg2
                elif op == "greater":
                    this_res = "yes" if arg1 > arg2 else "no"

                    
                # print("ind: ", ind)
                # print(this_res)
                res_dict[ind] = this_res


            elif "table" in op:
                table_dict = {}
                for row in table:
                    table_dict[row[0]] = row[1:]
                    
                if "#" in arg1:
                    arg1 = res_dict[int(arg1.replace("#", ""))]
                else:
                    if arg1 not in table_dict:
                        invalid_flag = 1
                        break
                    
                    cal_row = table_dict[arg1]
                    num_row = process_row(cal_row)
                    
                if num_row == "n/a":
                    invalid_flag = 1
                    break
                if op == "table_max":
                    this_res = max(num_row)
                elif op == "table_min":
                    this_res = min(num_row)
                elif op == "table_sum":
                    this_res = sum(num_row)
                elif op == "table_average":
                    this_res = sum(num_row) / len(num_row)
                    
                # this_res = round(this_res, 5)

                res_dict[ind] = this_res

            # print(this_res)

        if this_res != "yes" and this_res != "no" and this_res != "n/a":
            # print(this_res)
            this_res = round(this_res, 5)

    except:
        invalid_flag = 1
        

    return invalid_flag, this_res

def program_tokenization(original_program):
    original_program = original_program.split(', ')
    program = []
    for tok in original_program:
        cur_tok = ''
        for c in tok:
            if c == ')':
                if cur_tok != '':
                    program.append(cur_tok)
                    cur_tok = ''
            cur_tok += c
            if c in ['(', ')']:
                program.append(cur_tok)
                cur_tok = ''
        if cur_tok != '':
            program.append(cur_tok)
    program.append('EOF')
    return program

