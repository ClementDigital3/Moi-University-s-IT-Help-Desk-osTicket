# ICT Help Desk Ticket Management — task runner
# Run `make` on its own to see everything you can do.
PY := .venv/bin/python

.PHONY: help setup data prep models analysis chapters dashboard demo all \
        m1 m2 m3 end2end export clean-cache clean-results status

help:
	@echo ""
	@echo "  SETUP"
	@echo "    make setup        install dependencies from the vendored wheels"
	@echo "    make data         regenerate the synthetic corpus (skip if using real data)"
	@echo ""
	@echo "  PIPELINE            (make all runs everything in order)"
	@echo "    make prep         screening, cleaning, the shared train/test split"
	@echo "    make models       all four models"
	@echo "    make analysis     assemble tables, representation test, comparison, figures"
	@echo "    make chapters     write Chapters Four & Five into docs/"
	@echo "    make dashboard    build results/dashboard.html"
	@echo "    make all          prep -> models -> analysis -> chapters -> dashboard"
	@echo ""
	@echo "  ONE MODEL AT A TIME"
	@echo "    make end2end      Arm A, the end-to-end model"
	@echo "    make m1           classification   (must run before m2 and m3)"
	@echo "    make m2           resolver routing (needs m1)"
	@echo "    make m3           recommendation   (needs m1)"
	@echo ""
	@echo "  DEMO"
	@echo "    make export       persist all four models for serving"
	@echo "    make demo         serve all three views at http://127.0.0.1:8000/"
	@echo "                        /          Arm A live triage"
	@echo "                        /cascade   Arm B, the three models chained"
	@echo "                        /dashboard Chapter Four evidence"
	@echo ""
	@echo "  HOUSEKEEPING"
	@echo "    make status       what has been run, what is stale"
	@echo "    make clean-cache  drop cached embeddings and fitted models (forces a refit)"
	@echo ""

setup:
	$(PY) -m pip install --no-index --find-links=wheels \
	  scikit-learn xgboost pandas matplotlib sentence-transformers python-docx joblib

data:      ; $(PY) tools/generate_dataset.py
prep:      ; $(PY) -m analysis.preprocess
end2end:   ; $(PY) -m models.end_to_end.run
m1:        ; $(PY) -m models.classification.run
m2:        ; $(PY) -m models.routing.run
m3:        ; $(PY) -m models.recommendation.run
models:    end2end m1 m2 m3

analysis:
	$(PY) -m analysis.assemble_tables
	$(PY) -m analysis.representation
	$(PY) -m analysis.compare
	$(PY) -m analysis.figures

chapters:  ; $(PY) -m analysis.write_chapters
dashboard: ; $(PY) -m analysis.build_dashboard

export:
	$(PY) -m models.end_to_end.export
	$(PY) -m models.classification.export
	$(PY) -m models.routing.export
	$(PY) -m models.recommendation.export

demo: export
	@echo ""
	@echo "  /          Arm A live triage"
	@echo "  /cascade   Arm B cascade"
	@echo "  /dashboard evidence"
	@echo ""
	$(PY) -m models.end_to_end.serve

all: prep models analysis chapters dashboard
	@echo ""
	@echo "  done. dashboard: results/dashboard.html"
	@echo "        chapters : docs/"

status:
	@echo ""
	@printf "  %-34s %s\n" "corpus"           "$$([ -f data/processed/split_train.csv ] && echo ok || echo 'MISSING - make prep')"
	@printf "  %-34s %s\n" "end-to-end model" "$$([ -f results/tables/t43_armA_classification_routing.csv ] && echo ok || echo 'MISSING - make end2end')"
	@printf "  %-34s %s\n" "model 1 classification" "$$([ -f results/artifacts/classification.npz ] && echo ok || echo 'MISSING - make m1')"
	@printf "  %-34s %s\n" "model 2 routing"  "$$([ -f results/artifacts/routing.npz ] && echo ok || echo 'MISSING - make m2')"
	@printf "  %-34s %s\n" "model 3 recommendation" "$$([ -f results/artifacts/recommendation.npz ] && echo ok || echo 'MISSING - make m3')"
	@printf "  %-34s %s\n" "comparison"       "$$([ -f results/tables/t47_head_to_head.csv ] && echo ok || echo 'MISSING - make analysis')"
	@printf "  %-34s %s\n" "dashboard"        "$$([ -f results/dashboard.html ] && echo ok || echo 'MISSING - make dashboard')"
	@printf "  %-34s %s\n" "serving bundles"  "$$([ -f results/deploy/end_to_end.joblib ] && [ -f results/deploy/classification.joblib ] && [ -f results/deploy/routing.joblib ] && [ -f results/deploy/recommendation.joblib ] && echo 'ok (all four)' || echo 'MISSING - make export')"
	@echo ""

clean-cache:
	rm -rf results/cache results/deploy
	@echo "  caches dropped - the next run refits from scratch (slow)"

clean-results:
	rm -rf results/tables results/figures results/artifacts results/predictions
	@echo "  results dropped - run 'make all'"
