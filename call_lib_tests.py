# this python file tests all the methods in the call_lib.py file

from call_library import *


# do the tests

def test_get_team_id():

    # test with full name
    assert get_team_id('Boston Celtics') == '1610612738'

    # test with abbreviation
    assert get_team_id('BOS') == '1610612738'

    # test with wrong name
    assert get_team_id('Boston C') == None

    # test with wrong abbreviation
    assert get_team_id('BO') == None

    # test with wrong type
    assert get_team_id(1) == None

    # test with wrong type
    assert get_team_id(1.1) == None

    # test with wrong type
    assert get_team_id(True) == None

    # test with wrong type
    assert get_team_id(False) == None

    # test with wrong type
    assert get_team_id([1,2,3]) == None

    # test with wrong type
    assert get_team_id((1,2,3)) == None

    # test with wrong type
    assert get_team_id({'a':1, 'b':2}) == None

    # test with wrong type
    assert get_team_id(None) == None

    # test with wrong type
    assert get_team_id() == None

    print('test_get_team_id passed')



def test_get_team_name():

    # test with full name
    assert get_team_name('1610612738') == 'Boston Celtics'

    # test with abbreviation
    assert get_team_name('BOS') == 'Boston Celtics'

    # test with wrong name
    assert get_team_name('Boston C') == None

    # test with wrong abbreviation
    assert get_team_name('BO') == None

    # test with wrong type
    assert get_team_name(1) == None

    # test with wrong type
    assert get_team_name(1.1) == None

    # test with wrong type
    assert get_team_name(True) == None

    # test with wrong type
    assert get_team_name(False) == None

    # test with wrong type
    assert get_team_name([1,2,3]) == None

    # test with wrong type
    assert get_team_name((1,2,3)) == None

    # test with wrong type
    assert get_team_name({'a':1, 'b':2}) == None

    # test with wrong type
    assert get_team_name(None) == None

    # test with wrong type
    assert get_team_name() == None

    print('test_get_team_name passed')




def test_id_to_team_obj():

# test with full name
    assert id_to_team_obj('1610612738')[0]['full_name'] == 'Boston Celtics'

    # test with abbreviation
    assert id_to_team_obj('BOS')[0]['full_name'] == 'Boston Celtics'

    # test with wrong name
    assert id_to_team_obj('Boston C') == None

    # test with wrong abbreviation
    assert id_to_team_obj('BO') == None

    # test with wrong type
    assert id_to_team_obj(1) == None

    # test with wrong type
    assert id_to_team_obj(1.1) == None

    # test with wrong type
    assert id_to_team_obj(True) == None

    # test with wrong type
    assert id_to_team_obj(False) == None

    # test with wrong type
    assert id_to_team_obj([1,2,3]) == None

    # test with wrong type
    assert id_to_team_obj((1,2,3)) == None

    # test with wrong type
    assert id_to_team_obj({'a':1, 'b':2}) == None

    # test with wrong type
    assert id_to_team_obj(None) == None

    # test with wrong type
    assert id_to_team_obj() == None

    print('test_id_to_team_obj passed')



def run_all_tests():
    test_get_team_id()
    test_get_team_name()
    test_id_to_team_obj()
