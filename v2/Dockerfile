FROM jupyter/base-notebook:python-3.9.7

COPY ./ ./work

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]