import { Component, Input, Output, EventEmitter, OnInit, DoCheck } from '@angular/core';
import { trigger, state, style, transition, animate } from '@angular/core';
import { Observable } from 'rxjs/Observable';
import { ClassMetadataService } from '../class-metadata.service';

@Component({
    selector: 'element-editor',
    template: `
    <div *ngIf="classDef" class="element-container">
        <div *ngIf="name||_className" class="element-title" (click)="hideMe($event)">
            <span *ngIf="name">{{name}}</span>
            <span *ngIf="_className">{{_className}}</span>
        </div>
        <div class="element-body" [@bodyState]="bodyState">
            <template ngFor let-key [ngForOf]="classDefKeys">
                <div *ngIf="classDef[key].class==='sifra.modelling.structural.Info'">
                    <div class="string-display-name">
                        {{key}}
                    </div>
                    <div class="string-display-body">
                        {{classDef[key].value}}
                    </div>
                </div>
            </template>
            <template ngFor let-key [ngForOf]="classDefKeys">
                <template [ngIf]="key !== 'class' && classDef[key].class!=='sifra.modelling.structural.Info'">
                    <div [ngSwitch]="classDef[key].class">

                        <string-display *ngSwitchCase="'__builtin__.str'"
                            [name]="key"
                            [value]="classDef[key]"
                            (publish)="doPublish($event)">
                        </string-display>

                        <string-display *ngSwitchCase="'__builtin__.float'"
                            [name]="key"
                            [value]="classDef[key]"
                            [numeric]="true"
                            (publish)="doPublish($event)">
                        </string-display>

                        <string-display *ngSwitchCase="'__builtin__.int'"
                            [name]="key"
                            [value]="classDef[key]"
                            [numeric]="true"
                            (publish)="doPublish($event)">
                        </string-display>

                        <dict-display *ngSwitchCase="'__builtin__.dict'"
                            [name]="key"
                            (publish)="doPublish($event)">
                        </dict-display>

                        <pair-list-display *ngSwitchCase="'sifra.modelling.structures.XYPairs'"
                            [name]="key"
                            [value]="classDef[key]"
                            (publish)="doPublish($event)">
                        </pair-list-display>

                        <template ngSwitchDefault>
                            <div *ngIf="doThisBlock(classDef[key].class)">
                                <element-editor
                                    [name]="key"
                                    [value]="classDef[key].value || {}"
                                    (publish)="doPublish($event)">
                                </element-editor>
                            </div>
                        </template>

                    </div>
                </template>
            </template>
        </div>
    </div>
    `,
    styles: [`
        /deep/ .string-display-container {
            width: 100%;
        }

        /deep/ .string-display-container div {
            display: inline-block;
        }

        /deep/ .string-display-name {
            width: 200px;
            font-weight: bold;
            float: left;
        }

        .element-container {
            padding-left: 15px;
            background-color: #eee;
        }

        .element-container div {
            overflow: hidden;
        }

        .element-title {
            background-color: #00FFFF;
            padding-left: 3px;
            border: 1px solid #000000;
        }

        .element-body {
            border: 1px solid #000000;
        }
    `],
    animations: [
        trigger('bodyState', [
            state('invisible', style({
                height: 0,
                width: 0,
                visibility: 'hidden'
            })),
            state('visible', style({
                height: '*',
                width: '*',
                visibility: 'visible'
            })),
            transition('visible => invisible', animate('200ms ease-in')),
            transition('invisible => visible', animate('200ms ease-in'))
        ])
    ]
})
export class ElementEditorComponent implements OnInit, DoCheck {
    // The name of this element. Will be returned via publish.
    @Input() name: string = null;

    // Internal version of classname.
    _className: string = null;
    // The name of the class.
    @Input() className: string;

    // An instance of a document corresponding to the schema (classDef).
    private _value: any = {};

    // The definition of this element.
    @Input('value')
    set value(value: any) {
        this._value = value._value || {};
        delete value._value;
        this.classDef = value;
        this.getKeys();
    }
    get value(): any {
        return this._value;
    }

    // Fired when the value is modified.
    @Output() publish = new EventEmitter();

    // A list of available classes.
    availableClasses: string[] = [];

    // The 'schema' for the class.
    classDef: any = null;

    // Should the body be visible ("visible" for yes)?
    bodyState: string = "visible";

    classDefKeys: any = [];
    oldClassName: string = null;

    constructor(private classMetadataService: ClassMetadataService) {}

    ngOnInit() {
        if((this.className && this.classDef) || (!this.className && !this.classDef)) {
            throw new Error('strictly one of "classDef" and "className" must be provided');
        }

        this._className = this.className ? this.className : this.classDef.class;

        this.classMetadataService.getClassTypes().subscribe(
            availableClasses => {
                this.availableClasses = availableClasses;
                this.getKeys();
            },
            error => alert(<any>error)
        );

        if(this.className && !this.className.startsWith('__builtin__')) {
            let splits = this.className.split('.');
            let clsName = splits.pop();
            let modName = splits.join('.');
            this.value['class'] = [modName, clsName];
        }
    }

    ngDoCheck() {
        if(this.className && this.className != this.oldClassName) {
            this.classMetadataService.getClassDef(this.className).subscribe(
                classDef => {
                    this.classDef = classDef;
                    this.getKeys();
                },
                error => alert(<any>error)
            );
            this.oldClassName = this.className;
        }
    }

    doPublish($event) {
        this.value[$event.name] = $event.value;
        if(this.name) {
            this.publish.emit({name: this.name, value: this.value});
        } else {
            this.publish.emit(this.value);
        }
    }

    getKeys() {
        this.classDefKeys = this.classDef ? Object.keys(this.classDef) : [];
    }

    hideMe($event) {
        if(this.bodyState === "visible") {
            this.bodyState = "invisible";
        } else {
            this.bodyState = "visible";
        }
    }

    doThisBlock(clazz, fromSwitch) {
        return this.availableClasses && this.availableClasses.indexOf(clazz) > -1;
    }
}
