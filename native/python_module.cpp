#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "engine.hpp"
using ModelRef=std::shared_ptr<nd2::Model>;
struct Prepared {ModelRef owner;nd2::Context context;};
struct ReleaseGIL{PyThreadState* saved;ReleaseGIL():saved(PyEval_SaveThread()){}~ReleaseGIL(){PyEval_RestoreThread(saved);}};
static ModelRef& model(PyObject* capsule){
    auto p=static_cast<ModelRef*>(PyCapsule_GetPointer(capsule,"nd2.model"));if(!p)throw std::runtime_error("Invalid model handle");return *p;
}
static Prepared& prepared(PyObject* capsule){
    auto p=static_cast<Prepared*>(PyCapsule_GetPointer(capsule,"nd2.context"));if(!p)throw std::runtime_error("Invalid context handle");return *p;
}
static std::vector<int> ids(PyObject* obj){
    PyObject* seq=PySequence_Fast(obj,"Expected a sequence of integer token IDs");if(!seq)throw std::runtime_error("Invalid token sequence");
    auto n=PySequence_Fast_GET_SIZE(seq);if(n<1||n>32768){Py_DECREF(seq);throw std::runtime_error("Invalid token sequence length");}
    std::vector<int> out;out.reserve(size_t(n));
    for(Py_ssize_t i=0;i<n;++i){long v=PyLong_AsLong(PySequence_Fast_GET_ITEM(seq,i));
        if(PyErr_Occurred()||v<0||v>1000000){Py_DECREF(seq);throw std::runtime_error("Invalid token ID");}out.push_back(int(v));}
    Py_DECREF(seq);return out;
}
static PyObject* floats(const nd2::Vec& v){PyObject* list=PyList_New(Py_ssize_t(v.size()));if(!list)return nullptr;
    for(size_t i=0;i<v.size();++i){auto value=PyFloat_FromDouble(v[i]);if(!value){Py_DECREF(list);return nullptr;}PyList_SET_ITEM(list,Py_ssize_t(i),value);}return list;}
static PyObject* failure(const std::exception& e){PyErr_Clear();PyErr_SetString(PyExc_RuntimeError,e.what());return nullptr;}
static void destroy_model(PyObject* o){delete static_cast<ModelRef*>(PyCapsule_GetPointer(o,"nd2.model"));}
static void destroy_context(PyObject* o){delete static_cast<Prepared*>(PyCapsule_GetPointer(o,"nd2.context"));}
static PyObject* load(PyObject*,PyObject* args){const char* path;if(!PyArg_ParseTuple(args,"s",&path))return nullptr;
    try{ModelRef m;{ReleaseGIL release;m=std::make_shared<nd2::Model>(path);}return PyCapsule_New(new ModelRef(m),"nd2.model",destroy_model);}catch(const std::exception& e){return failure(e);}}
static PyObject* prepare(PyObject*,PyObject* args){PyObject *a,*b;if(!PyArg_ParseTuple(args,"OO",&a,&b))return nullptr;
    try{ModelRef m=model(a);auto v=ids(b);nd2::Context c;{ReleaseGIL release;std::lock_guard<std::mutex> lock(m->mutex);c=m->prepare(v);}
        return PyCapsule_New(new Prepared{m,std::move(c)},"nd2.context",destroy_context);}catch(const std::exception& e){return failure(e);}}
static PyObject* score(PyObject*,PyObject* args){PyObject *a,*b,*c;if(!PyArg_ParseTuple(args,"OOO",&a,&b,&c))return nullptr;
    try{ModelRef m=model(a);auto& ctx=prepared(b);nd2::require(ctx.owner.get()==m.get(),"Context belongs to a different model");
        PyObject* seq=PySequence_Fast(c,"Expected list of candidates");if(!seq)throw std::runtime_error("Invalid candidates");
        auto n=PySequence_Fast_GET_SIZE(seq);if(n<2||n>256){Py_DECREF(seq);throw std::runtime_error("Need 2..256 candidates");}
        std::vector<std::vector<int>> opts;
        try{for(Py_ssize_t i=0;i<n;++i)opts.push_back(ids(PySequence_Fast_GET_ITEM(seq,i)));}catch(...){Py_DECREF(seq);throw;}
        Py_DECREF(seq);nd2::Vec out;{ReleaseGIL release;std::lock_guard<std::mutex> lock(m->mutex);out=m->score(ctx.context,opts);}return floats(out);
    }catch(const std::exception& e){return failure(e);}}
static PyObject* encode(PyObject*,PyObject* args){PyObject *a,*b;if(!PyArg_ParseTuple(args,"OO",&a,&b))return nullptr;
    try{ModelRef m=model(a);auto v=ids(b);nd2::Vec out;{ReleaseGIL release;std::lock_guard<std::mutex> lock(m->mutex);out=m->encode(v);}return floats(out);}catch(const std::exception& e){return failure(e);}}
static PyObject* info(PyObject*,PyObject* args){PyObject* a;if(!PyArg_ParseTuple(args,"O",&a))return nullptr;
    try{auto m=model(a);return Py_BuildValue("{s:K,s:I}","mapped_bytes",(unsigned long long)m->mapped_bytes(),"hidden_size",m->cfg.d);}catch(const std::exception& e){return failure(e);}}
static PyObject* clear_cache(PyObject*,PyObject* args){PyObject* a;if(!PyArg_ParseTuple(args,"O",&a))return nullptr;
    try{auto m=model(a);{ReleaseGIL release;std::lock_guard<std::mutex> lock(m->mutex);m->clear_cache();}Py_RETURN_NONE;}catch(const std::exception& e){return failure(e);}}
static PyMethodDef methods[]={{"load",load,METH_VARARGS,"Memory-map model weights."},{"prepare",prepare,METH_VARARGS,"Encode one shared state."},
 {"score",score,METH_VARARGS,"Score a candidate set against shared state."},{"encode_ids",encode,METH_VARARGS,"Encoder output for numerical verification."},
 {"info",info,METH_VARARGS,"Runtime allocation metadata."},{"clear_cache",clear_cache,METH_VARARGS,"Clear candidate cache."},{nullptr,nullptr,0,nullptr}};
static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"_native","Standalone C++ decision runtime",-1,methods};
PyMODINIT_FUNC PyInit__native(void){return PyModule_Create(&module);}
