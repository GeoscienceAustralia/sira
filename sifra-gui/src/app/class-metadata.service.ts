import { Injectable } from '@angular/core';
import { Http, Request, Response, Headers, RequestOptions, RequestMethod } from '@angular/http';
import { Observable } from 'rxjs';
import 'rxjs/add/operator/catch';
import 'rxjs/add/operator/map';

@Injectable()
export class TestClassMetadataService {
    classDefs = {
        "sifra.modelling.elements.StepFunction": {
            "class": "sifra.modelling.elements.StepFunction",
            "description": {
                "class": "sifra.modelling.structural.Info",
                "value": "This is a description of the class"
            },
            "name": {
                "class": "__builtin__.string",
                "default": "Bloobaloob"
            },
            "xys": {
                "class": "sifra.modelling.structures.XYPairs"
            }
        },
        "sifra.modelling.component.Component": {
            "class": "sifra.modelling.component.Component",
            "description": {
                "class": "sifra.modelling.structural.Info",
                "value": "This is a cool component"
            },
            "FragilityCurve": {
                "class": "sifra.modelling.elements.StepFunction"
            }
        },
        "sifra.modelling.elements.Model": {
            "class": "sifra.modelling.elements.Model",
            "components": {
                "class": "__builtin__.dict",
                "description": {
                    "class": "sifra.modelling.structural.Info",
                    "value": "The components that make up the model"
                }
            }
        }
    };

    getClassTypes() {
        return Object.keys(this.classDefs);
    }

    getClassDef(className: string) {
        return this.classDefs[className];
    }

    getSubclassesOf(className: string) {
    }
}



@Injectable()
export class ClassMetadataService {
    SERVER_URL: string = window.location.protocol + '//' + window.location.hostname + ':5000';

    constructor(private http: Http) {
    }

    getTypeLists(hazard, sector, facility, component) {
        let url = this.SERVER_URL + '/sector-lists?' +
            'hazard=' + (hazard ? hazard : '') + '&' +
            'sector=' + (sector ? sector : '') + '&' +
            'facility_type=' + (facility ? facility : '') + '&' +
            'component=' + (component ? component : '');
        return this.makeAPIGetCall(url);
    }

    getClassTypes() {
        return this.makeAPIGetCall(this.SERVER_URL + '/class-types');
    }

    getClassDef(className: string) {
        return this.makeAPIGetCall(this.SERVER_URL + '/class-def/' + className);
    }

    getSubclassesOf(className: string) {
        return this.makeAPIGetCall(this.SERVER_URL + '/sub-classes-of/' + className);
    }

    getInstancesOf(className: string) {
        return this.makeAPIGetCall(this.SERVER_URL + '/instances-of/' + className);
    }

    getInstance(componentId: string) {
        return this.makeAPIGetCall(this.SERVER_URL + '/instance/' + componentId);
    }

    getCurrentModels() {
      return this.makeAPIGetCall(this.SERVER_URL + '/current_models');
    }

    getCurrentComponents() {
      return this.makeAPIGetCall(this.SERVER_URL + '/current_models');
    }

    save(data: any) {
        return this.makeAPIPostCall(this.SERVER_URL + '/save', data);
    }

    private makeAPIGetCall(url: string) {
        return this.http.get(url)
            .map(this.extractData)
            .catch(this.handleError);
    }

    private makeAPIPostCall(url: string, data: any) {
        let headers = new Headers({'Content-Type': 'application/json'});
        let requestOptions = new RequestOptions({
            method: RequestMethod.Post,
            url: url,
            headers: headers,
            body: data
        });
        return this.http.request(new Request(requestOptions))
            .map(this.extractData)
            .catch(this.handleError);
    }

    private extractData(res: Response) {
        return res.json();
    }

    private handleError (error: Response | any) {
        // In a real world app, we might use a remote logging infrastructure
        let errMsg: string;
        if (error instanceof Response) {
            const body = error.json() || '';
            const err = body.error || JSON.stringify(body);
            errMsg = `${error.status} - ${error.statusText || ''} ${err}`;
        } else {
            errMsg = error.message ? error.message : error.toString();
        }

        console.error(errMsg);
        return Observable.throw(errMsg);
    }
}
