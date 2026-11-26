"""The desktop interface.

    theme       palettes and the application stylesheet
    icons       the glyphs, drawn rather than shipped as files
    widgets     small controls the pages are built from
    transcript  the list of finished lines, and its model
    live        the card showing the sentence in progress
    settings    the settings page
    metrics     the performance panel
    window      the main window that holds them
    controller  what connects all of it to the pipeline

Nothing in here does any inference or audio work. The pipeline runs on its
own threads and reports through queued signals; a slow model can make a
caption late, but it cannot make a button unresponsive.
"""
