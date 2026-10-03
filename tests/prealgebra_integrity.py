"""Guard against turning incomplete worked examples into learning targets."""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from review_prealgebra_integrity import placeholder_description, worked_candidate


def check():
    problem = '<problem><para>Find the sum of the following two numbers: <m:math><m:mn>2</m:mn><m:mo>+</m:mo><m:mn>3</m:mn></m:math>.</para></problem>'
    solution = '<solution><para>The sum of two and three is five: <m:math><m:mn>5</m:mn></m:math>.</para></solution>'

    def inspect(body):
        return worked_candidate(ET.fromstring('<exercise xmlns="http://cnx.rice.edu/cnxml" '
            'xmlns:m="http://www.w3.org/1998/Math/MathML">' + body + '</exercise>'), 'Addition')

    result, reason = inspect(problem + solution)
    assert reason is None
    body = result[0].decode()
    assert body.count('Question:') == body.count('Answer:') == 1
    assert '2 + 3' in body and r'\(5\)' in body
    cases = [
        (problem, 'expected_one_problem_and_solution'),
        (problem + solution + solution, 'expected_one_problem_and_solution'),
        (problem + solution.replace('</para>', '<media alt="."><image src="unknown.png"/></media></para>'), 'requires_image_review'),
        (problem.replace('</para>', '<media alt="A complete-looking description"/></para>') + solution, 'requires_image_review'),
        (problem + solution.replace('</para>', '<image src="unknown.png"/></para>'), 'requires_image_review'),
        (problem.replace('</para>', '<link target-id="outside"/></para>') + solution, 'requires_cross_reference_review'),
        (problem + solution.replace('<m:mn>5</m:mn>', ''), 'conversion: Empty formula in extended source'),
        (problem + solution.replace('</para>', '<table><tgroup cols="1"><tbody><row><entry>First</entry><entry>Second</entry></row></tbody></tgroup></table></para>'),
         'conversion: Invalid or overlapping table span'),
    ]
    for source, expected in cases:
        result, reason = inspect(source)
        assert result is None and reason == expected, (reason, expected)
    assert all(placeholder_description(v) for v in ('', '.', '..', ' \t … '))
    assert not placeholder_description('0') and not placeholder_description('x + 1')
    selected = ET.fromstring('<exercise xmlns="http://cnx.rice.edu/cnxml" '
        'xmlns:m="http://www.w3.org/1998/Math/MathML">' + problem + solution + '</exercise>')
    selected.tail = 'EXTERNAL TAIL SENTINEL'
    isolated, reason = worked_candidate(selected, 'Addition')
    assert reason is None and b'EXTERNAL TAIL SENTINEL' not in isolated[0]
    return dict(passed=True, complete_pair_keeps_question_and_answer=True,
                incomplete_or_dependent_cases_rejected=len(cases),
                following_external_text_excluded=True,
                punctuation_placeholders_detected=True, short_numeric_description_preserved=True)


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
