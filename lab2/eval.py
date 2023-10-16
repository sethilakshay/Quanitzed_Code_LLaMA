import ast
import pandas as pd
# from timeout_decorator import timeout

df = pd.read_csv('./Results/output_7B_model.csv')
only_gen_code = list(df['only_gen_code'])
test_asserts = list(df['test_asserts'])


def check_asserts(only_gen_code, test_asserts):
    correct_code = 0
    idx = 0
    total_correct_asserts = 0
    total_asserts = 0
    for code, asserts in zip(only_gen_code, test_asserts):
        correct_asserts = 0
        try:
            # Execute the generated code
            asserts = ast.literal_eval(asserts)
            exec(code)
            for assertion in asserts:
                total_asserts+=1
                try:
                    # Parse the assertion
                    parsed_assertion = ast.parse(assertion)
                    # Extract left and right expressions
                    left_expression = ast.Expression(parsed_assertion.body[0].test.left)
                    right_expression = ast.Expression(parsed_assertion.body[0].test.comparators[0])
                    # Compile and evaluate the expressions
                    left_value = eval(compile(left_expression, filename="<ast>", mode="eval"))
                    right_value = eval(compile(right_expression, filename="<ast>", mode="eval"))
                    # Check if the assertion is True
                    if left_value == right_value:
                        correct_asserts += 1
                        total_correct_asserts+=1
                    else:
                        print(f"Assertion '{assertion}' failed.")
                except Exception as e:
                    print(f"Exception occurred: {e}")
        except Exception as e:
            print(f"Outer exception occurred: {e}")
        if correct_asserts==len(asserts):
            correct_code+=1
        idx+=1

    accuracy = (correct_code / len(only_gen_code)) * 100
    correct_assert_percentage = (total_correct_asserts/total_asserts)*100
    return accuracy, correct_assert_percentage

accuracy, correct_assert_percentage = check_asserts(only_gen_code, test_asserts)
print(f"Accuracy: {accuracy:.2f}%")
print(f"Correct Assert Percentage: {correct_assert_percentage:.2f}%")

