try:
    import pymysql
except ImportError:
    # PyMySQL is only needed when deploying against MySQL; local development
    # uses the SQLite backend configured in settings.py.
    pymysql = None
else:
    pymysql.install_as_MySQLdb()
