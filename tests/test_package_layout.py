def test_import_entrypoints():
    import api.app
    import api.worker_app
    import frontend.frontend_server
    import algo.model.miniseek_m.config

    assert api.app.app is not None
    assert api.worker_app.app is not None
    assert frontend.frontend_server.app is not None
