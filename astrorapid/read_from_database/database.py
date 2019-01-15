import sys
import os
import configparser
import pymysql
import getpass

# we should probably get these from a file instead
_MYSQL_CONFIG = {'host':'dlgreenmysqlv.stsci.edu',
                'port':43306,
                'user':'gnarayan',
                'database':'yse',
                'password':None}


def get_mysql_config():
    mysql_setting_file = os.path.expanduser(os.path.join('~','.my.cnf'))
    if os.path.exists(mysql_setting_file):
        reader = configparser.RawConfigParser()
        reader.read(mysql_setting_file)
        if 'client' in reader.sections():
            settings = dict(reader['client'].items())
            for key, old_val in _MYSQL_CONFIG.items():
                if old_val is None:
                    new_val = settings.get(key)
                    if new_val is not None:
                        _MYSQL_CONFIG[key] = new_val
    else:
        print("Create a ~/.my.cnf if you don't want to keep being bugged about the password.")
    return _MYSQL_CONFIG                


def make_mysql_schema_from_astropy_bintable_cols(astropy_columns):
    """
    Builds a schema for the data in the HEAD.fits file, converting from numpy
    types to mysql datatypes
    """
    fields = astropy_columns.names

    # the header files have some columns that are model dependent, so they
    # aren't present for all models. This eliminates those columns from the
    # mysql, so we are guaranteed to always have common columns for all files.
    # The price is that we will have to get these model specific columns from
    # the HEAD.fits, but it's less painful to do that for just a few objects
    # than to store several NULLs for all objects in mysql.
    use_fields = []
    for field in fields:
        if field.startswith('SIMSED')\
            or field.startswith('LCLIB')\
            or field.startswith('SIM_TEMPLATE')\
            or field.startswith('SIM_SALT2')\
            or field.startswith('SIM_GALFRAC'):
            continue
        else:
            use_fields.append(field)

    # upper case fieldnames are THE WORST.
    mysql_fields = ['objid',] + [field.lower().replace(')','').replace('(','_') for field in use_fields]

    # this is the FITS format codes
    # FITS format code         Description                     8-bit bytes
    # L                        logical (Boolean)               1
    # X                        bit                             *
    # B                        Unsigned byte                   1
    # I                        16-bit integer                  2
    # J                        32-bit integer                  4
    # K                        64-bit integer                  4
    # A                        character                       1
    # E                        single precision floating point 4
    # D                        double precision floating point 8
    # C                        single precision complex        8
    # M                        double precision complex        16
    # P                        array descriptor                8
    # Q                        array descriptor                16

    # we don't need to support all the FITS codes - just the subset that we're going to see in the data
    dtype_conversion = {'I':'INTEGER',
                        'J':'BIGINT',
                        'K':'BIGINT',
                        'A':'TINYTEXT',
                        'E':'FLOAT',
                        'D':'DOUBLE'}

    mysql_formats = ['VARCHAR(255)',] + [dtype_conversion.get(x[-1], 'TINYTEXT') for x in astropy_columns.formats]
    mysql_schema = ', '.join(['{} {}'.format(x, y) for x, y in zip(mysql_fields, mysql_formats)])
    return use_fields, mysql_fields, mysql_schema


def get_sql_password():
    """
    Prompts the user for a mysql password 
    Only accepts input from specified users
    """
    user = None
    try:
        user = getpass.getuser()
        password = getpass.getpass(prompt='Enter MySQL password: ', stream=sys.stderr)
        if user not in ['gnarayan', 'dmuthukrishna']:
            message = 'Unauthorized user {}'.format(user)
            raise RuntimeError(message)
        _MYSQL_CONFIG['password'] = password
    except Exception as e:
        message = '{}\nCould not get password from user {}'.format(e, user)
        raise RuntimeError(message)


def check_sql_db_for_table(table_name):
    """
    Check if a MySQL Table exists for this data_release
    """
    query = 'show tables'
    result = exec_sql_query(query)
    existing_tables = [table[0] for table in result]
    if table_name in existing_tables:
        return True 
    else:
        return False


def drop_sql_table_from_db(table_name):
    """
    Drops a MySQL Table from the table (assumes it exists)
    """
    query = 'drop table {}'.format(table_name)
    result = exec_sql_query(query)
    return result


def get_index_table_name_for_release(data_release):
    """
    Some datareleases are defined purely as ints which make invalid MySQL table names
    Fix that here by always prefixing release_ to the data_release name to make the MySQL table_name
    """
    table_name = 'release_{}'.format(data_release)
    return table_name


def create_sql_index_table_for_release(data_release, schema, redo=False, table_name=None):
    """
    Creates a MySQL Table to hold useful data from HEAD.fits files from the
    PLASTICC sim. Checks if the table exists already. The schema for this
    release must be supplied. This is not hardcoded since, other than SNID, we
    do not expect the schema to be the same for all data releases.
    Drops table if redo.
    Returns a table_name
    """
    if not table_name:
        table_name = get_index_table_name_for_release(data_release)
    result = check_sql_db_for_table(table_name)
    if result:
        print("Table {} exists.".format(table_name))
        if redo:
            print("Clobbering table {}.".format(table_name))
            drop_sql_table_from_db(table_name)
        else:
            return table_name

    query = 'CREATE TABLE {} ({}, PRIMARY KEY (objid))'.format(table_name, schema)
    result =  exec_sql_query(query)
    print("Created Table {}.".format(table_name))
    return table_name


def write_rows_to_index_table(index_entries, table_name):
    """
    Write rows to an index table
    Returns number of rows written
    """
    primitive = ['%s',]
    nrows = len(index_entries)
    if nrows == 0:
        return nrows

    ncols = len(index_entries[0])
    format_string = ', '.join(primitive*ncols)

    con = get_mysql_connection()
    query = f'INSERT INTO {table_name} VALUES ({format_string})'
    cursor = con.cursor()
    number_of_rows = cursor.executemany(query, index_entries)
    con.commit()
    con.close()
    return number_of_rows


def get_mysql_connection(attempts=0):
    """
    Get a  MySQL connection object. The config variable _MYSQL_CONFIG defines
    the context/parameters of the connection. If config does not include a
    MySQL password, the user is prompted for it.
    Returns a MySQL connection object.
    """
    _MYSQL_CONFIG = get_mysql_config() 
    password = _MYSQL_CONFIG.get('password')
    if password is None:
        get_sql_password()
    try:
        con = pymysql.connect(**_MYSQL_CONFIG)
    except Exception as e:
        attempts += 1
        if attempts < 3:
            _MYSQL_CONFIG.pop('password')
            con = get_mysql_connection(attempts=attempts)
        else:
            message = 'Login attempts exceeded!'
            raise RuntimeError(message)
    return con


def exec_big_sql_query(query, big=False):
    """
    Executes a supplied MySQL query. The context of the query is defined by
    _MYSQL_CONFIG and the connection object returned by get_mysql_connection() 
    Returns the result of the query (if any) or yields it as a generator if big=True
    """
    print(query)
    result = None
    con = get_mysql_connection()
    try:
        cursor = con.cursor()
        success = cursor.execute(query)
        print('Query results:',success)
        loop_ok = True 
        while loop_ok:
            result = cursor.fetchone()
            if result:
                yield result 
            else:
                loop_ok = False
    except Exception as e:
        message = '{}\nFailed to execute query\n{}'.format(e, query)
        raise RuntimeError(message)
    finally:
        con.close()
    return result

def exec_sql_query(query, big=False):
    """
    Executes a supplied MySQL query. The context of the query is defined by
    _MYSQL_CONFIG and the connection object returned by get_mysql_connection() 
    Returns the result of the query (if any) or yields it as a generator if big=True
    """
    print(query)
    result = None
    con = get_mysql_connection()
    try:
        cursor = con.cursor()
        success = cursor.execute(query)
        print('Query results:',success)
        result = cursor.fetchall()
    except Exception as e:
        message = '{}\nFailed to execute query\n{}'.format(e, query)
        raise RuntimeError(message)
    finally:
        con.close()
    return result
