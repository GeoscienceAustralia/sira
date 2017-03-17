import { Component, Input, Output, EventEmitter, OnInit, DoCheck } from '@angular/core';
import { trigger, state, style, transition, animate } from '@angular/core';
import { Observable } from 'rxjs/Observable';
import { ClassMetadataService } from '../class-metadata.service';

@Component({
    selector: 'element-editor',
    template: `
    <div *ngIf="_classDef" class="element-container">
        <div *ngIf="name||className" class="element-title" (click)="hideMe($event)">
            <span *ngIf="name">{{name}}</span>
            <span *ngIf="className">{{className}}</span>
        </div>
        <div class="element-body" [@bodyState]="bodyState">
            <template ngFor let-key [ngForOf]="classDefKeys">
                <div *ngIf="_classDef[key].class==='sifra.structural.Info'">
                    <div class="string-display-name">
                        {{key}}
                    </div>
                    <div class="string-display-body">
                        {{_classDef[key].value}}
                    </div>
                </div>
            </template>
            <template ngFor let-key [ngForOf]="classDefKeys">
                <template [ngIf]="key !== 'class' && _classDef[key].class!=='sifra.structural.Info'">
                    <div [ngSwitch]="_classDef[key].class">

                        <string-display *ngSwitchCase="'__builtin__.string'"
                            [name]="key"
                            [value]="_classDef[key]"
                            (publish)="doPublish($event)">
                        </string-display>

                        <string-display *ngSwitchCase="'__builtin__.float'"
                            [name]="key"
                            [value]="_classDef[key]"
                            [numeric]="true"
                            (publish)="doPublish($event)">
                        </string-display>

                        <dict-display *ngSwitchCase="'__builtin__.dict'"
                            [name]="key"
                            (publish)="doPublish($event)">
                        </dict-display>

                        <pair-list-display *ngSwitchCase="'sifra.structures.XYPairs'"
                            [name]="key"
                            [value]="_classDef[key]"
                            (publish)="doPublish($event)">
                        </pair-list-display>

                        <template ngSwitchDefault>
                            <div *ngIf="doThisBlock(_classDef[key].class)">
                                <element-editor
                                    [name]="key"
                                    [value]="_classDef[key]._value || {}"
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

    // The name of the class.
    @Input() className: string = null;

    // The definition of this element.
    @Input()
    set value(value: any) {
        this._value = value['_value'] || {};
        delete value['_value'];
        this._classDef = value;
        this.getKeys();
    }

    // Fired when the value is modified.
    @Output() publish = new EventEmitter();

    // A list of available classes.
    availableClasses: string[] = [];

    // The keys in this elements definition.
    private _classDef: any = null;
    private _value: any = {};
    classDefKeys: any = [];
    bodyState: string = "visible";
    oldClassName: string = null;

    constructor(private classMetadataService: ClassMetadataService) {}

    ngOnInit() {
        if((this.className && this._classDef) || (!this.className && !this._classDef)) {
            throw new Error('strictly one of "classDef" and "className" must be provided');
        }

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
            this._value['class'] = [modName, clsName];
        }
    }

    ngDoCheck() {
        if(this.className && this.className != this.oldClassName) {
            this.classMetadataService.getClassDef(this.className).subscribe(
                classDef => {
                    this._classDef = classDef;
                    this.getKeys();
                },
                error => alert(<any>error)
            );
            this.oldClassName = this.className;
        }
    }

    doPublish($event) {
        this._value[$event.name] = $event.value;
        if(this.name) {
            this.publish.emit({name: this.name, value: this._value});
        } else {
            this.publish.emit(this._value);
        }
    }

    getKeys() {
        this.classDefKeys = this._classDef ? Object.keys(this._classDef) : [];
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

