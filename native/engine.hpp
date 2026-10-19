#pragma once
// Standalone reference CPU inference. No libtorch, Python or ONNX dependency.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <filesystem>
#ifdef ND2_USE_BLAS
#include <cblas.h>
#endif
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
namespace nd2 {
using Vec = std::vector<float>;
inline void require(bool yes, const std::string& why) { if (!yes) throw std::runtime_error(why); }
struct MappedFile {
    const uint8_t* data=nullptr; size_t size=0;
#ifdef _WIN32
    HANDLE file=INVALID_HANDLE_VALUE, mapping=nullptr;
#else
    int fd=-1;
#endif
    explicit MappedFile(const std::string& path) {
        const uint32_t one=1; require(*reinterpret_cast<const uint8_t*>(&one)==1, "Little-endian host required");
#ifdef _WIN32
        auto wide=std::filesystem::u8path(path).wstring();
        file=CreateFileW(wide.c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
        if(file==INVALID_HANDLE_VALUE) throw std::runtime_error("Cannot open weights");
        LARGE_INTEGER n;
        if(!GetFileSizeEx(file,&n) || n.QuadPart<=0 || n.QuadPart>(int64_t(8)<<30)) {close();throw std::runtime_error("Invalid file size");}
        size=size_t(n.QuadPart); mapping=CreateFileMappingW(file,nullptr,PAGE_READONLY,0,0,nullptr);
        if(mapping) data=static_cast<const uint8_t*>(MapViewOfFile(mapping,FILE_MAP_READ,0,0,0));
#else
        fd=::open(path.c_str(),O_RDONLY);
        if(fd<0) throw std::runtime_error("Cannot open weights: "+path);
        struct stat st{};
        if(fstat(fd,&st)!=0 || st.st_size<=0 || uint64_t(st.st_size)>(uint64_t(8)<<30)) {close();throw std::runtime_error("Invalid file size");}
        size=size_t(st.st_size); auto p=mmap(nullptr,size,PROT_READ,MAP_PRIVATE,fd,0);
        if(p!=MAP_FAILED) data=static_cast<const uint8_t*>(p);
#endif
        if(!data) {close();throw std::runtime_error("Cannot map weights");}
    }
    void close() noexcept {
#ifdef _WIN32
        if(data) UnmapViewOfFile(data); if(mapping) CloseHandle(mapping);
        if(file!=INVALID_HANDLE_VALUE) CloseHandle(file);
        file=INVALID_HANDLE_VALUE; mapping=nullptr;
#else
        if(data) munmap(const_cast<uint8_t*>(data),size);
        if(fd>=0) ::close(fd); fd=-1;
#endif
        data=nullptr;
    }
    ~MappedFile(){close();}
    MappedFile(const MappedFile&)=delete; MappedFile& operator=(const MappedFile&)=delete;
};
struct Reader {
    const MappedFile& f; size_t pos=0;
    void need(size_t n){ require(pos<=f.size && n<=f.size-pos,"Truncated or malformed weights"); }
    template<class T> T read(){need(sizeof(T));T v;std::memcpy(&v,f.data+pos,sizeof(T));pos+=sizeof(T);return v;}
    std::string text(size_t n){need(n);std::string s(reinterpret_cast<const char*>(f.data+pos),n);pos+=n;return s;}
    void align(){size_t n=(64-pos%64)%64;need(n);pos+=n;}
};
struct Config {
    uint32_t vocab,d,ff,layers,heads,global_every,local_window,max_length,h,hops,pad;
    float eps,global_theta,local_theta;
};
struct Tensor {
    uint32_t rows=0,cols=0,rank=0,type=0;
    const float* fp=nullptr; const float* scales=nullptr; const int8_t* quant=nullptr;
    float at(size_t i) const {return type==0?fp[i]:float(quant[i])*scales[i/cols];}
};
inline Vec linear(const Vec& x,size_t rows,const Tensor& w,const Tensor* bias=nullptr){
    const size_t in=w.cols,out=w.rows;
    require(x.size()==rows*in,"Linear shape mismatch");
    Vec y(rows*out,0.0f);
#ifdef ND2_USE_BLAS
    if(w.type==0) {
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,int(rows),int(out),int(in),1.0f,
                    x.data(),int(in),w.fp,int(in),0.0f,y.data(),int(out));
    } else {
        // Only a small weight tile is dequantized, never the complete matrix.
        const size_t block=64; Vec temp(block*in);
        for(size_t o=0;o<out;o+=block){size_t n=std::min(block,out-o);
            for(size_t j=0;j<n;++j) for(size_t k=0;k<in;++k) temp[j*in+k]=w.at((o+j)*in+k);
            cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,int(rows),int(n),int(in),1.0f,
                        x.data(),int(in),temp.data(),int(in),0.0f,y.data()+o,int(out));
        }
    }
#else
    // Portable reference kernel. Not an optimized INT8 GEMM implementation.
    for(size_t r=0;r<rows;++r) for(size_t o=0;o<out;++o){
        float sum=0;
        if(w.type==0){const float* wp=w.fp+o*in;for(size_t k=0;k<in;++k)sum+=x[r*in+k]*wp[k];}
        else {const int8_t* wp=w.quant+o*in;for(size_t k=0;k<in;++k)sum+=x[r*in+k]*float(wp[k]);sum*=w.scales[o];}
        y[r*out+o]=sum;
    }
#endif
    if(bias)for(size_t r=0;r<rows;++r)for(size_t o=0;o<out;++o)y[r*out+o]+=bias->at(o);
    return y;
}
inline Vec norm(const Vec& x,size_t rows,size_t d,const Tensor& gamma,float eps,const Tensor* beta=nullptr){
    Vec y(x.size());
    for(size_t r=0;r<rows;++r){double mean=0,variance=0;
        for(size_t j=0;j<d;++j)mean+=x[r*d+j];mean/=d;
        for(size_t j=0;j<d;++j){double a=x[r*d+j]-mean;variance+=a*a;}
        float inv=float(1/std::sqrt(variance/d+eps));
        for(size_t j=0;j<d;++j)y[r*d+j]=(x[r*d+j]-float(mean))*inv*gamma.at(j)+(beta?beta->at(j):0);
    } return y;
}
inline float gelu(float x){return 0.5f*x*(1.0f+std::erf(x*0.7071067811865475f));}
inline Vec softmax(const Vec& x){
    require(!x.empty(),"Empty probability vector");
    for(float v:x)require(std::isfinite(v),"Non-finite model output");
    float m=*std::max_element(x.begin(),x.end());double sum=0;Vec p(x.size());
    for(size_t i=0;i<x.size();++i){p[i]=std::exp(x[i]-m);sum+=p[i];}
    for(float& v:p)v/=float(sum);return p;
}
struct Context {Vec keys,values;size_t length=0;};
class Model {
    MappedFile file;
    std::unordered_map<std::string,Tensor> tensors;
    struct CacheEntry {std::vector<int> ids;Vec vector;};
    std::list<CacheEntry> cache;
public:
    Config cfg{};
    std::mutex mutex;
    explicit Model(const std::string& path):file(path){
        Reader r{file};require(r.text(8)==std::string("NDC2BIN\0",8),"Bad model magic");
        require(r.read<uint32_t>()==1,"Unsupported model version");
        uint32_t* vals[]={&cfg.vocab,&cfg.d,&cfg.ff,&cfg.layers,&cfg.heads,&cfg.global_every,
            &cfg.local_window,&cfg.max_length,&cfg.h,&cfg.hops,&cfg.pad};
        for(auto p:vals)*p=r.read<uint32_t>();
        cfg.eps=r.read<float>();cfg.global_theta=r.read<float>();cfg.local_theta=r.read<float>();
        require(cfg.vocab>=8&&cfg.vocab<=1000000&&cfg.d>=8&&cfg.d<=2048&&cfg.ff>=8&&cfg.ff<=8192,
                "Invalid encoder dimensions");
        require(cfg.layers>=1&&cfg.layers<=64&&cfg.heads>=1&&cfg.heads<=64&&cfg.d%cfg.heads==0&&(cfg.d/cfg.heads)%2==0,
                "Invalid attention configuration");
        require(cfg.global_every>0&&cfg.global_every<=64&&cfg.local_window>=2&&cfg.local_window<=8192,
                "Invalid attention window");
        require(cfg.max_length>=8&&cfg.max_length<=32768&&cfg.h>=8&&cfg.h<=2048&&cfg.hops>=1&&cfg.hops<=8&&cfg.pad<cfg.vocab,
                "Invalid model configuration");
        require(std::isfinite(cfg.eps)&&cfg.eps>0&&std::isfinite(cfg.global_theta)&&cfg.global_theta>0&&
                std::isfinite(cfg.local_theta)&&cfg.local_theta>0,"Invalid numeric configuration");
        uint32_t count=r.read<uint32_t>();require(count>0&&count<=4096,"Invalid tensor count");
        for(uint32_t i=0;i<count;++i){
            uint32_t name_len=r.read<uint32_t>();require(name_len>0&&name_len<512,"Invalid tensor name");
            std::string name=r.text(name_len);Tensor t;
            t.rank=r.read<uint32_t>();t.rows=r.read<uint32_t>();t.cols=r.read<uint32_t>();t.type=r.read<uint32_t>();
            uint64_t nbytes=r.read<uint64_t>();
            require((t.rank==1||t.rank==2)&&t.rows>0&&t.rows<=1000000&&t.cols>0&&t.cols<=16384&&(t.rank!=1||t.cols==1)&&t.type<=1,
                    "Invalid tensor metadata");
            require(t.type==0||t.rank==2,"Only matrices can be quantized");
            uint64_t elements=uint64_t(t.rows)*t.cols;
            uint64_t expected=t.type==0?elements*4:elements+uint64_t(t.rows)*4;
            require(nbytes==expected&&nbytes<=file.size,"Invalid tensor byte count");
            r.align();r.need(size_t(nbytes));
            if(t.type==0)t.fp=reinterpret_cast<const float*>(file.data+r.pos);
            else {t.scales=reinterpret_cast<const float*>(file.data+r.pos);t.quant=reinterpret_cast<const int8_t*>(file.data+r.pos+t.rows*4);}
            r.pos+=size_t(nbytes);require(tensors.emplace(name,t).second,"Duplicate tensor name");
        }
        // Validate every used tensor before running native pointer arithmetic.
        check("encoder.embedding.weight",cfg.vocab,cfg.d);check("encoder.embed_norm.weight",cfg.d);
        check("encoder.final_norm.weight",cfg.d);
        for(uint32_t i=0;i<cfg.layers;++i){auto p="encoder.layers."+std::to_string(i)+".";
            if(i)check(p+"attn_norm.weight",cfg.d);
            check(p+"qkv.weight",3*cfg.d,cfg.d);check(p+"attn_out.weight",cfg.d,cfg.d);
            check(p+"mlp_norm.weight",cfg.d);check(p+"mlp_in.weight",2*cfg.ff,cfg.d);check(p+"mlp_out.weight",cfg.d,cfg.ff);
        }
        for(auto n:{"key","value","candidate"})check(std::string("head.")+n+".weight",cfg.h,cfg.d);
        for(uint32_t i=0;i<cfg.hops;++i){auto s=std::to_string(i);
            check("head.updates."+s+".weight",cfg.h,2*cfg.h);check("head.updates."+s+".bias",cfg.h);
            check("head.norms."+s+".weight",cfg.h);check("head.norms."+s+".bias",cfg.h);
        }
        check("head.local.weight",cfg.h,4*cfg.h);check("head.local.bias",cfg.h);
        check("head.set_layer.weight",cfg.h,3*cfg.h);check("head.set_layer.bias",cfg.h);
        check("head.output.weight",1,cfg.h);check("head.output.bias",1);
    }
    const Tensor& t(const std::string& name)const{
        auto it=tensors.find(name);require(it!=tensors.end(),"Missing tensor "+name);return it->second;
    }
    void check(const std::string& name,uint32_t rows,uint32_t cols=0)const{
        const auto& a=t(name);require(a.rows==rows&&a.cols==(cols?cols:1)&&a.rank==(cols?2u:1u),"Wrong tensor shape "+name);
    }
    void check_ids(const std::vector<int>& ids)const{
        require(!ids.empty()&&ids.size()<=cfg.max_length,"Token sequence is empty or exceeds maximum length");
        for(int id:ids)require(id>=0&&uint32_t(id)<cfg.vocab,"Token ID out of vocabulary");
    }
    Vec encode(const std::vector<int>& ids)const{
        check_ids(ids);size_t n=ids.size(),d=cfg.d,hd=d/cfg.heads;
        const auto& emb=t("encoder.embedding.weight");Vec x(n*d);
        for(size_t i=0;i<n;++i)for(size_t j=0;j<d;++j)x[i*d+j]=emb.at(size_t(ids[i])*d+j);
        x=norm(x,n,d,t("encoder.embed_norm.weight"),cfg.eps);
        for(size_t layer=0;layer<cfg.layers;++layer){
            std::string p="encoder.layers."+std::to_string(layer)+".";
            Vec pre=layer?norm(x,n,d,t(p+"attn_norm.weight"),cfg.eps):x;
            Vec qkv=linear(pre,n,t(p+"qkv.weight"));
            bool local=layer%cfg.global_every!=0;
            float theta=local?cfg.local_theta:cfg.global_theta;
            for(size_t pos=0;pos<n;++pos)for(size_t h=0;h<cfg.heads;++h)for(size_t j=0;j<hd/2;++j){
                float freq=std::pow(theta,-float(2*j)/float(hd));float phase=float(pos)*freq;
                float co=std::cos(phase),si=std::sin(phase);
                for(size_t part=0;part<2;++part){size_t base=pos*3*d+part*d+h*hd;
                    float a=qkv[base+j],b=qkv[base+j+hd/2];
                    qkv[base+j]=a*co-b*si;qkv[base+j+hd/2]=b*co+a*si;
                }
            }
            Vec attended(n*d,0.0f),row(n);const float scale=1/std::sqrt(float(hd));
            // Attention is evaluated row-by-row, no L x L probability buffer.
            for(size_t h=0;h<cfg.heads;++h)for(size_t qi=0;qi<n;++qi){
                size_t lo=local?(qi>cfg.local_window/2?qi-cfg.local_window/2:0):0;
                size_t hi=local?std::min(n,qi+cfg.local_window/2+1):n;
                float maximum=-std::numeric_limits<float>::infinity();
                for(size_t ki=lo;ki<hi;++ki){float s=0;
                    for(size_t j=0;j<hd;++j)s+=qkv[qi*3*d+h*hd+j]*qkv[ki*3*d+d+h*hd+j];
                    row[ki]=s*scale;maximum=std::max(maximum,row[ki]);
                }
                double den=0;for(size_t ki=lo;ki<hi;++ki){row[ki]=std::exp(row[ki]-maximum);den+=row[ki];}
                for(size_t ki=lo;ki<hi;++ki){float a=row[ki]/float(den);
                    for(size_t j=0;j<hd;++j)attended[qi*d+h*hd+j]+=a*qkv[ki*3*d+2*d+h*hd+j];
                }
            }
            Vec a=linear(attended,n,t(p+"attn_out.weight"));for(size_t i=0;i<x.size();++i)x[i]+=a[i];
            pre=norm(x,n,d,t(p+"mlp_norm.weight"),cfg.eps);
            Vec both=linear(pre,n,t(p+"mlp_in.weight")),g(n*cfg.ff);
            for(size_t i=0;i<n;++i)for(size_t j=0;j<cfg.ff;++j)g[i*cfg.ff+j]=gelu(both[i*2*cfg.ff+j])*both[i*2*cfg.ff+cfg.ff+j];
            a=linear(g,n,t(p+"mlp_out.weight"));for(size_t i=0;i<x.size();++i)x[i]+=a[i];
        }
        return norm(x,n,d,t("encoder.final_norm.weight"),cfg.eps);
    }
    Context prepare(const std::vector<int>& ids)const{
        Vec x=encode(ids);return {linear(x,ids.size(),t("head.key.weight")),linear(x,ids.size(),t("head.value.weight")),ids.size()};
    }
    Vec candidate(const std::vector<int>& ids){
        for(auto it=cache.begin();it!=cache.end();++it)if(it->ids==ids){Vec v=it->vector;cache.splice(cache.begin(),cache,it);return v;}
        Vec x=encode(ids),pooled(cfg.d,0.0f);
        for(size_t i=0;i<ids.size();++i)for(size_t j=0;j<cfg.d;++j)pooled[j]+=x[i*cfg.d+j]/float(ids.size());
        Vec v=linear(pooled,1,t("head.candidate.weight"));
        // Candidate cache only, not full context or results. Bounded entries + IDs.
        if(ids.size()<=1024){cache.push_front({ids,v});if(cache.size()>256)cache.pop_back();}
        return v;
    }
    Vec score(const Context& ctx,const std::vector<std::vector<int>>& options){
        require(options.size()>=2&&options.size()<=256,"Need 2..256 options");
        size_t c=options.size(),h=cfg.h,n=ctx.length;
        require(n>0&&ctx.keys.size()==n*h&&ctx.values.size()==n*h,"Invalid prepared context");
        Vec original(c*h),q;
        for(size_t i=0;i<c;++i){Vec v=candidate(options[i]);std::copy(v.begin(),v.end(),original.begin()+i*h);}
        q=original;Vec row(n);
        for(size_t hop=0;hop<cfg.hops;++hop){Vec joined(c*2*h);
            for(size_t i=0;i<c;++i){
                for(size_t j=0;j<n;++j){float v=0;for(size_t k=0;k<h;++k)v+=q[i*h+k]*ctx.keys[j*h+k];row[j]=v/std::sqrt(float(h));}
                Vec prob=softmax(row);
                for(size_t k=0;k<h;++k){float a=0;for(size_t j=0;j<n;++j)a+=prob[j]*ctx.values[j*h+k];
                    joined[i*2*h+k]=q[i*h+k];joined[i*2*h+h+k]=a;}
            }
            auto s=std::to_string(hop);Vec update=linear(joined,c,t("head.updates."+s+".weight"),&t("head.updates."+s+".bias"));
            for(size_t i=0;i<q.size();++i)q[i]+=gelu(update[i]);
            q=norm(q,c,h,t("head.norms."+s+".weight"),cfg.eps,&t("head.norms."+s+".bias"));
        }
        Vec features(c*4*h);
        for(size_t i=0;i<c;++i)for(size_t k=0;k<h;++k){float a=q[i*h+k],b=original[i*h+k];
            features[i*4*h+k]=a;features[i*4*h+h+k]=b;features[i*4*h+2*h+k]=a*b;features[i*4*h+3*h+k]=std::abs(a-b);}
        Vec u=linear(features,c,t("head.local.weight"),&t("head.local.bias"));for(float& v:u)v=gelu(v);
        Vec mean(h,0),maximum(h,-std::numeric_limits<float>::infinity());
        for(size_t i=0;i<c;++i)for(size_t k=0;k<h;++k){mean[k]+=u[i*h+k]/float(c);maximum[k]=std::max(maximum[k],u[i*h+k]);}
        Vec set(c*3*h);for(size_t i=0;i<c;++i)for(size_t k=0;k<h;++k){set[i*3*h+k]=u[i*h+k];set[i*3*h+h+k]=mean[k];set[i*3*h+2*h+k]=maximum[k];}
        Vec v=linear(set,c,t("head.set_layer.weight"),&t("head.set_layer.bias"));for(float& a:v)a=gelu(a);
        return linear(v,c,t("head.output.weight"),&t("head.output.bias"));
    }
    void clear_cache(){cache.clear();}
    size_t mapped_bytes()const{return file.size;}
};
} // namespace nd2
